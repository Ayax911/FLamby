import numpy as np
from sklearn import metrics


def metric(y_true, logits):
    """Balanced accuracy over the 2 classes (robust to the benign/malignant imbalance,
    e.g. kau-bcmd has 6 malignant out of 222 test images)."""
    y_true = y_true.reshape(-1)
    preds = np.argmax(logits, axis=1)
    return metrics.balanced_accuracy_score(y_true, preds)
