import math

import torch
import torch.nn.functional as F

from fairseq.criterions.ctc_multi_loss import CTCMultiLoss, _CTCMultiLoss, CTCEncoderWrapperModel, CTCFakeEncoderModel, CTCFakeDecoderModel
from fairseq import utils
from fairseq.criterions import register_criterion

@register_criterion("rdrop_ctc_multi_loss")
class RDropCTCMultiLoss(CTCMultiLoss):

    @staticmethod
    def add_args(parser):
        CTCMultiLoss.add_args(parser)
        parser.add_argument('--rdrop-weight', default=0, type=int,
                            help='The relative weight to assign to the rdrop loss')

    @classmethod
    def build_criterion(cls, args, task):
        underlying_criterion = RDropCTCMultiLoss.build_underlying_criterion(args, task)
        return _RDropCTCMultiLoss(args, task, underlying_criterion)

class _RDropCTCMultiLoss(_CTCMultiLoss):
    def __init__(self, args, task, underlying_criterion):
        super().__init__(args, task, underlying_criterion)
        self.rdrop_weight = args.rdrop_weight

    def forward_train(self, model, sample, optimizer, ignore_grad, reduce=True, log_probs=True):
        sample_input = sample['net_input']
        sample_concat_input = {
            'src_tokens': torch.cat([sample_input['src_tokens'], sample_input['src_tokens'].clone()], 0),
            'src_lengths': torch.cat([sample_input['src_lengths'], sample_input['src_lengths'].clone()], 0),
            'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
        }
        encoder_target = torch.cat([sample["encoder_target"], sample["encoder_target"].clone()], 0)
        decoder_target = torch.cat([sample["target"], sample["target"].clone()], 0)
        encoder_sample = {
            "net_input": sample_concat_input,
            "target": encoder_target,
            "target_lengths": torch.cat([sample["encoder_target_lengths"], sample["encoder_target_lengths"].clone()], 0),
            "ntokens": sum(torch.cat([sample["encoder_target_lengths"], sample["encoder_target_lengths"].clone()], 0)).item()
        }
        decoder_sample = {
            "net_input": sample_concat_input,
            "target": decoder_target,
            "target_lengths": torch.cat([sample["target_lengths"], sample["target_lengths"].clone()], 0),
            "ntokens": sum(torch.cat([sample["target_lengths"], sample["target_lengths"].clone()], 0)).item()
        }
        decoder_pad_mask = sample["target"].unsqueeze(-1).eq(self.padding_idx)
                
        decoder_out, encoder_out = self.ctc_aware_model(model, **sample_concat_input)
        encoder_fake_model = CTCFakeEncoderModel(model.encoder, encoder_out, encoder_target)
        decoder_fake_model = CTCFakeDecoderModel(model, decoder_out, decoder_target)
        ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion(encoder_fake_model, encoder_sample, reduce=reduce, log_probs=log_probs)
        real_loss, _, real_logging_output = self.underlying_criterion(decoder_fake_model, decoder_sample, reduce=reduce)
        kl_loss = self.compute_kl_loss(decoder_fake_model, decoder_out, decoder_pad_mask) 
        loss = self.ctc_weight * ctc_loss + real_loss + self.rdrop_weight * kl_loss

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

    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
        
        p_loss = F.kl_div(p, q_tec, reduction='none')
        q_loss = F.kl_div(q, p_tec, reduction='none')
        
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

