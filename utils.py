"""
Helper functions for project.
"""

import numpy as np
from tensorflow.random import uniform

def get_diff(A,B):
    """
    Get the difference between elements of two lists.
    """
    return [a-b for a,b in zip(A,B)]

def add_noise(labels, C):
    """
    Randomly chooses between 0 and 1 for each label with probabilities C.
    """
    return np.stack([np.random.choice(2, p=C[label]) for label in labels])

def simulate_label_noise(labels, total_batches, ind, perc, C, seed=10):
    X = uniform([total_batches], seed=seed)
    l = 0

    if X[ind] > (1-perc):
       labels = add_noise(labels, C)
       l = 1
    
    return labels, l  

def flip_labels_C(corruption_prob, num_classes, seed=1):
    """
    Transition matrix.
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)

    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    
    return C
