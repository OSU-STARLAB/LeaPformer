# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion


@dataclass
class RDropLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    rdrop_weight: int = field(
        default=0,
        metadata={"help": "The relative weight to assign to the rdrop loss"},
    )

@register_criterion("rdrop_label_smoothed_cross_entropy", dataclass=RDropLabelSmoothedCrossEntropyCriterionConfig)
class RDropLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=True, rdrop_weight=0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.rdrop_weight = rdrop_weight

    def forward_train(self, model, sample, optimizer, ignore_grad, reduce=True):
        sample_input = sample['net_input']
        sample_concat_input = {
            'src_tokens': torch.cat([sample_input['src_tokens'], sample_input['src_tokens'].clone()], 0),
            'src_lengths': torch.cat([sample_input['src_lengths'], sample_input['src_lengths'].clone()], 0),
            'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
        }
        sample_concat_target = torch.cat([sample["target"], sample["target"].clone()], 0)
        sample_concat = {
            "net_input": sample_concat_input,
            "target": sample_concat_target,
            "target_lengths": torch.cat([sample["target_lengths"], sample["target_lengths"].clone()], 0),
            "ntokens": sum(torch.cat([sample["target_lengths"], sample["target_lengths"].clone()], 0)).item()
        }
        pad_mask = sample["target"].unsqueeze(-1).eq(self.padding_idx)

        net_output = model(**sample_concat_input)
        loss, nll_loss = self.compute_loss(model, net_output, sample_concat, reduce=reduce)
        
        kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
        loss += self.rdrop_weight * kl_loss

        sample_size = (sample_concat["target"].size(0) if self.sentence_avg else sample_concat["ntokens"])
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample_concat["ntokens"],
            "nsentences": sample_concat["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

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


