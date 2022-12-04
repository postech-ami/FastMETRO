"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Basic logger. It Computes and stores the average and current value
"""

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class EvalMetricsLogger(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch = 0
        # define a upper-bound performance (worst case) 
        # numbers are in unit millimeter
        self.mPVPE = 100.0/1000.0
        self.mPJPE = 100.0/1000.0
        self.PAmPJPE = 100.0/1000.0
        
    def set(self, epoch=0, mPVPE=0.1, mPJPE=0.1, PAmPJPE=0.1):
        self.epoch = epoch
        self.mPVPE = mPVPE
        self.mPJPE = mPJPE
        self.PAmPJPE = PAmPJPE

    def update(self, mPVPE, mPJPE, PAmPJPE, epoch):
        self.epoch = epoch
        self.mPVPE = mPVPE
        self.mPJPE = mPJPE
        self.PAmPJPE = PAmPJPE
        
