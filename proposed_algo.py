"""
Implementation of self-aware SGD.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Local imports
from Oracle import Oracle_Phase1
from Model import get_predictions
from utils import get_diff, simulate_label_noise, flip_labels_C


def self_aware_SGD(model, W, train_loader, val_loader, loss_fn, opt, name, epochs=50, b=0.5, p=0.8, tau=1e-7, noisy=True):
    """
    `b`:     batch probability for noise simulation
    `p`:     example probability for noise simulation
    `tau`:   threshold for rewards
    `noisy`: whether to simulate label noise
    """

    W1 = model.get_weights()
    diff_grad = get_diff(W, W1)  # historic gradient
    weights_loc = f'./models/{name}.h5'

    C = flip_labels_C(p, 2)  # transition matrix to simulate label noise

    # bandit training
    oracle = get_trained_bandit(
        model, train_loader, val_loader, loss_fn, diff_grad, tau, b, C)

    # training begins
    Training_loss = []
    val_aucs = []
    best_val_auc = 0  # best val score

    for epoch in range(epochs):
        loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.numpy()
            targets = targets.numpy()

            # if we want to simulate noise
            if noisy:
                targets, _ = simulate_label_noise(
                    targets, len(train_loader), i, b, C)

            with tf.GradientTape() as tape1:
                p = model.forward(inputs, train=True)[:, 0]
                l = loss_fn(p, targets)

            # computing gradients for this batch
            grad = tape1.gradient(l, model.trainable_variables)

            # vectorising historic grad
            M = tf.concat([tf.reshape(x, [-1]) for x in diff_grad], axis=0)
            G = tf.concat([tf.reshape(x, [-1])
                          for x in grad], axis=0)      # vectorising gradient

            grad_norm = tf.norm(G)
            cosine = tf.reduce_sum(G*M)/(tf.norm(G)*tf.norm(M))

            A = np.array([grad_norm, cosine])
            A = np.expand_dims(A, axis=0)  # input feature to oracle/bandit
            pred = oracle.predict(A)

            if pred < 0.4:  # oracle output 0 implies gradient consistency. This threshold could be a hyperparameter. We are keeping it fixed.
                opt.apply_gradients(zip(grad, model.trainable_variables))
                W1 = model.get_weights()
                # updating historic gradient to reflect the recent changes
                diff_grad = get_diff(W, W1)

            loss += 1

        Training_loss.append(loss/(i+1))
        Preds, Labels = get_predictions(model, val_loader)
        val_auc = roc_auc_score(Labels, Preds)
        val_aucs.append(val_auc)

        # checkpoint to store best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save_weights(weights_loc)

        print(f"Epoch: {epoch:.1f} \nVal Auc: {val_auc:.5f}")

    model.load_weights(weights_loc)  # load the best model config

    return model


def get_trained_bandit(model, train_loader, val_loader, loss_fn, diff_grad, tau, b, C):

    opt = tf.keras.optimizers.SGD()
    M = tf.concat([tf.reshape(x, [-1]) for x in diff_grad],
                  axis=0)  # historic gradient vectorisation
    W = model.get_weights()

    Features = []
    Reward = []

    # compute performance of intial model on val data
    P, L = get_predictions(model, val_loader)
    prev = roc_auc_score(L, P)

    # analyse impact of each batch on initial model's perf to compute reward
    for i, data in tqdm(enumerate(train_loader)):

        inputs, targets = data
        inputs = inputs.numpy()
        targets = targets.numpy()
        targets, _ = simulate_label_noise(targets, len(train_loader), i, b, C)  # if we want to simulate noise

        with tf.GradientTape() as tape1:
            p = model.forward(inputs, train=False)[:, 0]
            l = loss_fn(p, targets)

        grad = tape1.gradient(l, model.trainable_variables)

        # compute features to the oracle/bandit model
        G = tf.concat([tf.reshape(x, [-1]) for x in grad], axis=0)
        grad_norm = tf.norm(G)
        cosine = tf.reduce_sum(G*M)/(tf.norm(G)*tf.norm(M))
        A = np.array([grad_norm, cosine])  # representing this gradient update
        # append to list, to be used later for training bandit model
        Features.append(A)

        # impact of applying gradient to the initial model
        opt.apply_gradients(zip(grad, model.trainable_variables))
        P, L = get_predictions(model, val_loader)
        cur = roc_auc_score(L, P)
        Reward.append(cur-prev)

        model.set_weights(W)  # revert to the initial state

    F = np.array(Features)
    R = np.array(Reward)

    # get index of batches that had negative impact, hence are noisy
    ind_neg = np.where(R < (-1*tau))[0]
    F1 = F[ind_neg, :]
    print(ind_neg.shape)
    # R1 = R[ind_neg]

    # get index of batches that had positive impact, hence are not noisy
    ind_pos = np.where(R > tau)[0]
    F0 = F[ind_pos, :]
    print(ind_pos.shape)

    # prepare data for bandit training
    FF = np.concatenate((F1, F0))
    labels = np.concatenate((np.ones(len(F1)), np.zeros(len(F0))))

    # train oracle with multiple initial states till we get the satisfiable training loss
    for _ in range(0, 10):
        oracle = Oracle_Phase1()
        oracle.build((None, 2))
        oracle.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanAbsoluteError())
        check = tf.keras.callbacks.ModelCheckpoint(
            './oracle.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
        hist = oracle.fit(x=FF, y=labels, batch_size=512,
                          epochs=2500, callbacks=[check], shuffle=True)

        if hist.history['loss'][-1] < 0.2:
            break

    oracle.load_weights('./oracle.h5')  # load the stored weights

    return oracle
