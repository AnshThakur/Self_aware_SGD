"""
Bandit model.
"""

from tensorflow.keras import layers, Model


class Oracle_Phase1(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.d1 = layers.Dense(8, activation='relu')
        self.d2 = layers.Dense(4, activation='relu')
        self.d3 = layers.Dense(1, activation='sigmoid')

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

    def call(self, inputs):
        x = self.forward(inputs)
        return x
