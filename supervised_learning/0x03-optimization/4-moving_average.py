#!/usr/bin/env python3
'Moving Average'


def moving_average(data, beta):
    'calculates the weighted moving average of a data set'
    vt = 0
    average = []
    for v in range(len(data)):
        vt = beta * vt + (1-beta) * data[v]
        bias = 1 - beta ** (v+1)
        average.append(vt/bias)
    return average
