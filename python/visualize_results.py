# python/visualize_results.py

import matplotlib.pyplot as plt


def plot_accuracy(baseline_acc, pruned_acc, int8_acc, out_path):
    labels = ["Baseline", "Pruned", "INT8"]
    values = [baseline_acc, pruned_acc, int8_acc]
    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
