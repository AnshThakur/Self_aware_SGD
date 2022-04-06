#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers

class Oracle_Phase1(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.d1 = layers.Dense(8, activation='relu')
        self.d2 = layers.Dense(4, activation='relu')
        self.d3 = layers.Dense(1, activation='sigmoid')

    # `train` kwarg has no effect
    def forward(self, x,train=False):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)     
        return x

    def call(self, inputs):
        x = self.forward(inputs,train=False)
        return x 
