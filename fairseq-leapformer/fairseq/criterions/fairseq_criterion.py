# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.nn.modules.loss import _Loss


class FairseqCriterion(_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare(self, samples):
        """Prepare criterion for DataParallel training."""
        raise NotImplementedError

    def forward(self, net_output, sample):
        """Compute the loss for the given sample and network output."""
        raise NotImplementedError

    def aggregate(self, losses):
        """Aggregate losses from DataParallel training.

        Takes a list of losses as input (as returned by forward) and
        aggregates them into the total loss for the mini-batch.
        """
        raise NotImplementedError
