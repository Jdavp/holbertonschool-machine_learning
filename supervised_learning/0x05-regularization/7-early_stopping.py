#!/usr/bin/env python3
'Early Stopping'
import tensorflow as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    'determines if you should stop gradient descent early'
    if (opt_cost-cost) > threshold:
        count = 0
    else:
        count += 1
    if patience == count:
        return True, count
    else:
        return False, count
