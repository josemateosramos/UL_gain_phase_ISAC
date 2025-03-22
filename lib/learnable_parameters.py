# -*- coding: utf-8 -*-
import torch.nn as nn

# Impairment learning parameters
class ImpairmentNet(nn.Module):
    '''
    Class that contains the gain-phase impairments to learn.
    Inputs to the constructor:
        - init_vector: initial estimate of the gain-phase impairments.
    '''
    def __init__(self,init_vector):
        super(ImpairmentNet, self).__init__()
        #Define vector to optimize
        self.impairments = nn.Parameter(init_vector)