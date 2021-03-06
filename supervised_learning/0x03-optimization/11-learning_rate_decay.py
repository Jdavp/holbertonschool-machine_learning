#!/usr/bin/env python3
"Learning Rate Decay"


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    'updates the learning rate using inverse time decay in numpy'

    epoch = global_step // decay_step
    alpha = alpha/(1+decay_rate * epoch)

    return alpha
