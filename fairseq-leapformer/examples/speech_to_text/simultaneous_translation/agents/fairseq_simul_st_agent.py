import math
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks
from fairseq.file_io import PathManager

import copy

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift:
        ]

        torch.manual_seed(1)
        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        ).numpy()

        output = self.transform(output)

        return torch.from_numpy(output)

    def transform(self, input):
        if self.global_cmvn is not None:
            mean = self.global_cmvn["mean"]
            std = self.global_cmvn["std"]
        
            x = np.subtract(input, mean)
            x = np.divide(x, std)
        else:
            mean = x.mean(axis=0)
            square_sums = (x**2).sum(axis=0)
            x = np.subtract(x, mean)
            var = square_sums / x.shape[0] - mean**2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):

        if len(self.value) == 0:
            self.value = value
            return

        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class FairseqSimulSTAgent(SpeechAgent):

    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab(args)

        if getattr(
            self.model.decoder.layers[0].encoder_attn,
            'pre_decision_ratio',
            None
        ) is not None:
            self.speech_segment_size *= (
                self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
            )

        args.global_cmvn = None
        if args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        if args.global_stats:
            with PathManager.open(args.global_stats, "r") as f:
                global_cmvn = json.loads(f.read())
                self.global_cmvn = {"mean": global_cmvn["mean"], "std": global_cmvn["stddev"]}

        self.feature_extractor = OnlineFeatureExtractor(args)

        self.max_len = args.max_len
        self.max_len_after_finish_read = args.max_len_after_finish_read
        print(f"Max len overall: {self.max_len}, max len after finish reading: {self.max_len_after_finish_read}", flush=True)

        self.force_finish = args.force_finish
        
        self.flush_method = args.flush_method
        print(f"Flush method set to: {self.flush_method}", flush=True)

        self.leapformer_attn_enable = args.leapformer_attn_enable
        if self.leapformer_attn_enable:
            self.simul_attn_chkpts = {}

        torch.set_grad_enabled(False)

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--max-len-after-finish-read", type=int, default=25,
                            help="Max length of translation after finished reading")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--waitk", type=int, default=None,
                            help="Wait-k delay for evaluation")
        parser.add_argument("--flush-method", type=str, default="none",
                            help="Method used to flush state after each sentence and enable more continuous operation.")
        parser.add_argument("--leapformer-attn-enable", default=False, action="store_true",
                            help="Enable Leapformer attention, assuming provided model has Leapformer attention.")

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config is not None:
            task_args.config_yaml = args.config

        if args.waitk is not None:
            state["cfg"]["model"].waitk_lagging = args.waitk
            self.waitk_lagging = args.waitk
        else:
            self.waitk_lagging = state["cfg"]["model"].waitk_lagging
        
        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        print(self.model)

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()

        # initialize leapformer recurrent state dictionary 
        if self.leapformer_attn_enable:
            model_size = len(self.model.encoder.transformer_layers) + len(self.model.decoder.layers)
            self.simul_attn_chkpts["layers"] = {}
            for i in range(model_size):
                self.simul_attn_chkpts["layers"][i] = {}
                self.simul_attn_chkpts["layers"][i]["self_attn"] = {}
                self.simul_attn_chkpts["layers"][i]["cross_attn"] = {}

                self.simul_attn_chkpts["layers"][i]["self_attn"]["norm_sin"] = None
                self.simul_attn_chkpts["layers"][i]["self_attn"]["norm_cos"] = None
                self.simul_attn_chkpts["layers"][i]["self_attn"]["k_sin"] = None
                self.simul_attn_chkpts["layers"][i]["self_attn"]["k_cos"] = None
                self.simul_attn_chkpts["layers"][i]["self_attn"]["kTv_sin"] = None
                self.simul_attn_chkpts["layers"][i]["self_attn"]["kTv_cos"] = None
                
                # while we can save some cross_attn values, for simultaneous 
                # environments it isn't actually that efficient to do so
                # when the encoder isn't unidirectional (i.e. K and V are entirely
                # changing at each time step versus changing progressively)

                # leaving here for illustrative purposes or if unidirectional
                # encoders are set up in the future (non-trivial due to acoustic
                # boundaries on downsampling)

                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["norm_sin"] = None
                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["norm_cos"] = None
                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["k_sin"] = None
                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["k_cos"] = None
                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["kTv_sin"] = None
                # self.simul_attn_chkpts["layers"][i]["cross_attn"]["kTv_cos"] = None

            self.simul_attn_chkpts["old_indices"] = {
                "src": 0,
                "tgt": 0,
            }

        self.past_finish_read = 0

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # Merge sub word to full word.
        
        if None in units.value:
            units.value.remove(None)

        # Check for special conditions
        if len(units) < 1:
            return None
        elif (
            (self.model.decoder.dictionary.eos() == units[-1])
            or (('<e>' == self.model.decoder.dictionary.string([units[-1]])) and states.finish_read())
            or (len(states.units.target) > self.max_len)
            or (self.past_finish_read > self.max_len_after_finish_read)
        ):
            tokens = [self.model.decoder.dictionary.string([unit]) for unit in units]
            for j in range(len(units)):
                units.pop()
            return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

        # Regular handling if no special conditions
        segment = []
        for index in units:
            token = self.model.decoder.dictionary.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        return None

    def update_model_encoder(self, states):
        print(f"States.units.source len: {len(states.units.source)}", flush=True)
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths, encoder_inference_flag=True)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            if states.finish_read():
                return WRITE_ACTION
            else:
                return READ_ACTION

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [x for x in states.units.target.value if x is not None]
            ).unsqueeze(0)
        )

        states.incremental_states["steps"] = {
            "src": states.encoder_states["encoder_out"][0].size(0),
            "tgt": 1 + len(states.units.target),
        }

        states.incremental_states["online"] = {"only": torch.tensor(not states.finish_read())}

        # required because of quirks with simultaneous architecture in fairseq, likely
        # due to requirements for other inference modes (i.e. schedulers)
        if self.leapformer_attn_enable:
            simul_chkpts_dc = copy.deepcopy(self.simul_attn_chkpts["layers"][0])
            x, outputs = self.model.decoder.forward(
                prev_output_tokens=tgt_indices,
                encoder_out=states.encoder_states,
                incremental_state=states.incremental_states,
                simul_attn_chkpts=self.simul_attn_chkpts,
            )
            self.simul_attn_chkpts["old_indices"] = states.incremental_states["steps"]

        else:
            x, outputs = self.model.decoder.forward(
                prev_output_tokens=tgt_indices,
                encoder_out=states.encoder_states,
                incremental_state=states.incremental_states,
            )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        if (outputs.action == 0) and (not states.finish_read()):
            if self.leapformer_attn_enable:
                self.simul_attn_chkpts["layers"][0] = simul_chkpts_dc
            return READ_ACTION
        else:
            if states.finish_read():
                self.past_finish_read += 1
                print(f"Past finish: {self.past_finish_read}", flush=True)

                # LeaPformer NOTE: after finishing our reads (i.e. no encoder changes),
                #                  we can finally save our key and value matrices
                #                  for free, no additional comp. on KV states, leaving
                #                  this here for illustrative purposes

            return WRITE_ACTION

    def predict(self, states):
        if not getattr(states, "encoder_states", None):
            return self.model.decoder.dictionary.eos()
        
        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        index = index[0, 0].item()

        print(f"Incremental predicted output: {self.model.decoder.dictionary.string([index])}", flush=True)

        if (
            self.force_finish
            and index == self.model.decoder.dictionary.eos()
            and not states.finish_read()
        ):
            # If we want to force finish the translation
            # (don't stop before finish reading), return a None
            # self.model.decoder.clear_cache(states.incremental_states)
            index = None

        return index

    def flush(self, states):
        if (self.flush_method is not None) and (self.flush_method != "no_flush"):
            flush_method = self.flush_method
            
            #print(f"Source len before flush: {len(states.units.source)}", flush=True)
            if states.finish_read():
                # Force finish translation since, assuming we are keeping up with translation, there is no additional sentence to translate
                states.units.source = TensorListEntry()
                states.encoder_states = None
            else:
                # For all methods, take into account pre_decision_ratio & downsampling factor from conv)
                if flush_method == 'naive':
                    # Naively flush entire source (except one segment for correctness)
                    flush_amount = 1 * 4 * self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
                    states.units.source.value = states.units.source.value[-flush_amount:]
                    self.update_states_read(states)
                elif flush_method == 'keep_last_k':    
                    # Flush up to last k elements of source
                    # Roughly the amount that needs to be translated if we still have additional audio to process
                    flush_amount = self.waitk_lagging * 4 * self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
                    if flush_amount >= len(states.units.source): # Edge case, don't keep everything, instead remove first pre_decision_ratio elements to prevent stall
                        print(f"Flush amount: {self.model.decoder.layers[0].encoder_attn.pre_decision_ratio}", flush=True)
                        states.units.source.value = states.units.source.value[self.model.decoder.layers[0].encoder_attn.pre_decision_ratio:]
                    else:
                        print(f"Flush amount: {flush_amount}", flush=True)
                        states.units.source.value = states.units.source.value[-flush_amount:]
                    self.update_states_read(states)
                elif flush_method == 'decoder_sync':    
                    # Flush number of elements from source that have been translated by decoder
                    #print(f"Target units: {states.units.target}", flush=True)
                    flush_amount = int( 0.75 * len(states.units.target)  * 4 * self.model.decoder.layers[0].encoder_attn.pre_decision_ratio )
                    print(f"Flush amount: {flush_amount}", flush=True)
                    states.units.source.value = states.units.source.value[flush_amount:]

                    if len(states.units.source) >= ( 4 * self.model.decoder.layers[0].encoder_attn.pre_decision_ratio):
                        self.update_states_read(states)
                    else:
                        states.encoder_states = None
            #print(f"Source len after flush: {len(states.units.source)}", flush=True)
            
            states.units.target = ListEntry()
            states.incremental_states = dict()
