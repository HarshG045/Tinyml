# python/early_exit_tuner.py

import numpy as np
import tensorflow as tf


def compute_conf_threshold(exit1_model: tf.keras.Model,
                           x_val,
                           y_val,
                           target_accuracy: float = 0.95,
                           min_threshold: float = 0.5,
                           max_threshold: float = 0.99,
                           steps: int = 50):
    """
    Find a confidence threshold such that among samples above threshold,
    the accuracy is at least target_accuracy.
    """
    preds = exit1_model.predict(x_val, verbose=0)
    confs = np.max(preds, axis=1)
    pred_labels = np.argmax(preds, axis=1)
    correct = pred_labels == y_val

    best_th = min_threshold
    for th in np.linspace(min_threshold, max_threshold, steps):
        mask = confs >= th
        if np.sum(mask) == 0:
            continue
        acc = np.mean(correct[mask])
        if acc >= target_accuracy:
            best_th = float(th)
            break

    return best_th
