import math

import torch
from torch import nn
import torch.nn.functional as F


from fairseq.criterions.CTC_loss import CTCCriterion
from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import BaseFairseqModel


@register_criterion("ctc_multi_loss")
class CTCMultiLoss(FairseqCriterion):
    
    @staticmethod
    def build_underlying_criterion(args, task):
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @staticmethod
    def add_args(parser):
        CTCCriterion.add_args(parser)
        parser.add_argument('--ctc-weight', default=1.0, type=float, metavar='W',
                            help='The relative weight to assign to the CTC loss')
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the model output loss')
    
    @classmethod
    def build_criterion(cls, args, task):
        underlying_criterion = CTCMultiLoss.build_underlying_criterion(args, task)
        return _CTCMultiLoss(args, task, underlying_criterion)

class CTCEncoderWrapperModel(BaseFairseqModel):
    def forward(self, model, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = model.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=False)
        decoder_out = model.decoder(prev_output_tokens, encoder_out=encoder_out)

        ctc_features = encoder_out["ctc_out"][0]
        encoder_padding_mask = encoder_out["ctc_padding_mask"]
        if encoder_padding_mask != []:
            encoder_padding_mask = encoder_padding_mask[0].t()  # B x T => T x B

        return decoder_out, {
            "encoder_out": ctc_features,
            "encoder_padding_mask": encoder_padding_mask
        }

class CTCFakeEncoderModel(nn.Module):
    def __init__(self, encoder, net_out, target):
        super().__init__()
        self.net_out = net_out
        self.target = target

    def forward(self, **unused):
        return self.net_out

    def get_targets(self, *unused):
        return self.target

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                probs = F.log_softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            return probs
        raise NotImplementedError

class CTCFakeDecoderModel(nn.Module):
    def __init__(self, model, net_out, target):
        super().__init__()
        self.model = model
        self.net_out = net_out
        self.target = target

    def forward(self, **unused):
        return self.net_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.model.get_normalized_probs(net_output, log_probs, sample=sample)
            
    def get_targets(self, *unused):
        return self.target

    @property
    def decoder(self):
        return self.model.decoder

class _CTCMultiLoss(FairseqCriterion):
    def __init__(self, args, task, underlying_criterion):
        super().__init__(task)
        assert task.source_dictionary is not None
        self.ctc_aware_model = CTCEncoderWrapperModel()
        self.ctc_criterion = CTCCriterion(args, task)
        self.underlying_criterion = underlying_criterion
        self.ctc_weight = args.ctc_weight

    def forward(self, model, sample, reduce=True, log_probs=True):
        decoder_out, encoder_out = self.ctc_aware_model(model, **sample["net_input"])
        encoder_fake_model = CTCFakeEncoderModel(model.encoder, encoder_out, sample["encoder_target"])
        decoder_fake_model = CTCFakeDecoderModel(model, decoder_out, sample["target"])
        encoder_sample = {
            "net_input": sample["net_input"],
            "target": sample["encoder_target"],
            "target_lengths": sample["encoder_target_lengths"],
            "ntokens": sum(sample["encoder_target_lengths"]).item()
        }
        ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion(
            encoder_fake_model, encoder_sample, reduce=reduce, log_probs=log_probs)
        real_loss, _, real_logging_output = self.underlying_criterion(
            decoder_fake_model, sample, reduce=reduce)
        loss = self.ctc_weight * ctc_loss + real_loss

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ctc_loss": ctc_logging_output['loss'],
            "ntokens": real_logging_output['ntokens'],
            "nsentences": real_logging_output['nsentences'],
            "sample_size": real_logging_output['sample_size'],
            "ctc_errors": ctc_logging_output['errors'],
            "ctc_total": ctc_logging_output['total'],
            "nframes": ctc_logging_output['nframes'],
        }
        if 'nll_loss' in real_logging_output:
            logging_output['nll_loss'] = real_logging_output['nll_loss']
        return loss, ctc_sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

    @staticmethod
    def reduce_metrics(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get('ctc_loss', 0) for log in logging_outputs))
        if logging_outputs and 'nll_loss' in logging_outputs[0]:
            nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        else:
            nll_loss_sum = loss_sum - ctc_loss_sum  # NLL computed on the real loss, not on the auxiliary CTC
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        ctc_errors = sum(log.get("ctc_errors", 0) for log in logging_outputs)
        ctc_total = sum(log.get("ctc_total", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('ctc_loss', ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ctc_acc', 100.0 - min(ctc_errors * 100.0 / ctc_total, 100.0), round=2)
        metrics.log_scalar('nframes', nframes, round=2)
        
