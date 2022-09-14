# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch

from fairseq.tasks import register_task
from fairseq.tasks.speech_text_joint import SpeechTextJointToTextTask
from fairseq.optim.amp_optimizer import AMPOptimizer

logger = logging.getLogger(__name__)


@register_task("rdrop_speech_text_joint_to_text")
class RDropSpeechTextJointToTextTask(SpeechTextJointToTextTask):

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion.forward_train(model, sample, optimizer, ignore_grad)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

