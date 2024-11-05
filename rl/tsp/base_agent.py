#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
import random
import numpy as np


class Agent(object):
    def __init__(self):
        pass

    def expand_state_vector(self,state):
        if len(state.shape) == 1 or len(state.shape)==3:
            return np.expand_dims(state,axis = 0) #add a dimation for batch_size
        else:
            return state

    def remember(self,*args):
        self.memory.save(args)





