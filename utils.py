"""
Helper functions for project.
"""

import numpy as np
import tensorflow as tf

def get_diff(A,B):
    difference = []
    zip_object = zip(A, B)
    
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)

    return difference      


def add_noise(labels,C):
    """
    Randomly chooses between 0 and 1 for each label 
    with probabilities from C.
    """
    L = [np.random.choice(2, p=C[label]) for label in labels]
    return np.stack(L)


def simulate_label_noise(labels,total_batches,ind,perc,C):
    X = tf.random.uniform([total_batches], minval=0, maxval=1, seed=10)
    l = 0
    if X[ind] > (1-perc):
       labels = add_noise(labels,C)
       l = 1
    return labels, l  

## transition matrix
def flip_labels_C(corruption_prob, num_classes, seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)

    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C
