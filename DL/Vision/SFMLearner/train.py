# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from options import MonodepthOptions
from trainer import Trainer

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":

    trainer = Trainer(opts)
    trainer.train()
