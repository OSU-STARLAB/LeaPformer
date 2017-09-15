# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from contextlib import ExitStack
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from fairseq import utils


class SequenceGenerator(object):
    def __init__(self, models, dst_dict, beam_size=1, minlen=1, maxlen=200,
                 stop_early=True, normalize_scores=True, len_penalty=1):
        """Generates translations of a given source sentence.

        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.models = models
        self.dict = dst_dict
        self.pad = dst_dict.pad()
        self.eos = dst_dict.eos()
        self.vocab_size = len(dst_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.positions = torch.LongTensor(range(self.pad + 1, self.pad + maxlen + 2))
        self.decoder_context = models[0].decoder.context_size()
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty

    def cuda(self):
        for model in self.models:
            model.cuda()
        self.positions = self.positions.cuda()
        return self

    def generate_batched_itr(self, data_itr, maxlen_a=0, maxlen_b=200,
                             cuda_device=None, timer=None):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda_device: GPU on which to do generation.
            timer: StopwatchMeter for timing generations.
        """

        def lstrip_pad(tensor):
            return tensor[tensor.eq(self.pad).sum():]

        for sample in data_itr:
            s = utils.prepare_sample(sample, volatile=True, cuda_device=cuda_device)
            input = s['net_input']
            srclen = input['src_tokens'].size(1)
            if timer is not None:
                timer.start()
            hypos = self.generate(input['src_tokens'], input['src_positions'],
                                  maxlen=(maxlen_a*srclen + maxlen_b))
            if timer is not None:
                timer.stop(s['ntokens'])
            for i, id in enumerate(s['id']):
                src = input['src_tokens'].data[i, :]
                # remove padding from ref, which appears at the beginning
                ref = lstrip_pad(s['target'].data[i, :])
                yield id, src, ref, hypos[i]

    def generate(self, src_tokens, src_positions, beam_size=None, maxlen=None):
        """Generate a batch of translations."""
        with ExitStack() as stack:
            for model in self.models:
                stack.enter_context(model.decoder.incremental_inference())
            return self._generate(src_tokens, src_positions, beam_size, maxlen)

    def _generate(self, src_tokens, src_positions, beam_size=None, maxlen=None):
        bsz = src_tokens.size(0)
        beam_size = beam_size if beam_size is not None else self.beam_size
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        encoder_outs = []
        for model in self.models:
            model.eval()
            model.decoder.clear_incremental_state()  # start a fresh sequence

            # compute the encoder output and expand to beam size
            encoder_out = model.encoder(src_tokens, src_positions)
            encoder_out = self._expand_encoder_out(encoder_out, beam_size)
            encoder_outs.append(encoder_out)

        # initialize buffers
        scores = encoder_outs[0][0].data.new(bsz * beam_size).fill_(0)
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        align = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(-1)
        align_buf = align.clone()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': float('Inf')} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz)*beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}
        def buffer(name, type_of=tokens):
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                bbsz = sent*beam_size
                best_unfinalized_score = scores[bbsz:bbsz+beam_size].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                scores: A vector of the same size as bbsz_idx containing scores
                    for each hypothesis
            """
            assert bbsz_idx.numel() == scores.numel()
            norm_scores = scores/math.pow(step+1, self.len_penalty) if self.normalize_scores else scores
            sents_seen = set()
            for idx, score in zip(bbsz_idx.cpu(), norm_scores.cpu()):
                sent = idx // beam_size
                sents_seen.add(sent)

                def get_hypo():
                    hypo = tokens[idx, 1:step+2].clone()
                    hypo[step] = self.eos
                    alignment = align[idx, 1:step+2].clone()
                    return {
                        'tokens': hypo,
                        'score': score,
                        'alignment': alignment,
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            # return number of hypotheses finished this step
            num_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        reorder_state = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                for model in self.models:
                    model.decoder.reorder_incremental_state(reorder_state)

            probs, avg_attn_scores = self._decode(tokens[:, :step+1], encoder_outs)
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
            else:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores.view(-1, 1))

            # record alignment to source tokens, based on attention
            _ignore_scores = buffer('_ignore_scores', type_of=scores)
            avg_attn_scores.topk(1, out=(_ignore_scores, align[:, step+1].unsqueeze(1)))

            # take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            probs.view(bsz, -1).topk(cand_size, out=(cand_scores, cand_indices))
            torch.div(cand_indices, self.vocab_size, out=cand_beams)
            cand_indices.fmod_(self.vocab_size)

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                eos_bbsz_idx = buffer('eos_bbsz_idx')
                cand_bbsz_idx.masked_select(eos_mask, out=eos_bbsz_idx)
                if eos_bbsz_idx.numel() > 0:
                    eos_scores = buffer('eos_scores', type_of=scores)
                    cand_scores.masked_select(eos_mask, out=eos_scores)
                    num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add((eos_mask*cand_size).type_as(cand_offsets), cand_offsets,
                      out=active_mask)

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            active_mask.topk(beam_size, 1, largest=False, out=(_ignore, active_hypos))
            active_bbsz_idx = buffer('active_bbsz_idx')
            cand_bbsz_idx.gather(1, active_hypos, out=active_bbsz_idx)
            active_scores = cand_scores.gather(1, active_hypos,
                                               out=scores.view(bsz, beam_size))

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # finalize all active hypotheses once we hit maxlen
            # finalize_hypos will take care of adding the EOS markers
            if step == maxlen:
                num_remaining_sent -= finalize_hypos(step, active_bbsz_idx, active_scores)
                assert num_remaining_sent == 0
                break

            # copy tokens for active hypotheses
            torch.index_select(tokens[:, :step+1], dim=0, index=active_bbsz_idx,
                               out=tokens_buf[:, :step+1])
            cand_indices.gather(1, active_hypos,
                                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step+1])

            # copy attention/alignment for active hypotheses
            torch.index_select(align[:, :step+2], dim=0, index=active_bbsz_idx,
                               out=align_buf[:, :step+2])

            # swap buffers
            old_tokens = tokens
            tokens = tokens_buf
            tokens_buf = old_tokens
            old_align = align
            align = align_buf
            align_buf = old_align

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(bsz):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_outs):
        length = tokens.size(1)

        # repeat the first length positions to fill batch
        positions = self.positions[:length].view(1, length)

        # wrap in Variables
        tokens = Variable(tokens, volatile=True)
        positions = Variable(positions, volatile=True)

        avg_probs = None
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            decoder_out, attn = model.decoder(tokens, positions, encoder_out)
            probs = F.softmax(decoder_out[:, -1, :]).data
            attn = attn[:, -1, :].data
            if avg_probs is None or avg_attn is None:
                avg_probs = probs
                avg_attn = attn
            else:
                avg_probs.add_(probs)
                avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()
        avg_attn.div_(len(self.models))

        return avg_probs, avg_attn

    def _expand_encoder_out(self, encoder_out, beam_size):
        res = []
        for tensor in encoder_out:
            res.append(
                # repeat beam_size times along second dimension
                tensor.repeat(1, beam_size, *[1 for i in range(tensor.dim()-2)]) \
                # then collapse into [bsz*beam, ...original dims...]
                .view(-1, *tensor.size()[1:])
            )
        return tuple(res)
