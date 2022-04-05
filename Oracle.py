#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:46:22 2021

@author: engs2258
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score
from torch.utils.data import Dataset
import torch
import random


class Oracle_Phase1(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.d1=layers.Dense(8,activation='relu')
        self.d2=layers.Dense(4,activation='relu')
        self.d3=layers.Dense(1,activation='sigmoid')


    def forward(self, x,train=False):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)     
        return x

    def call(self, inputs):
        x = self.forward(inputs,train=False)
        return x 
    

