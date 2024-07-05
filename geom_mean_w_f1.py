from river import metrics
import math

class GeomMeanWeightedF1(metrics.base.MultiClassMetric):
    """Geometric Mean Weighted F1 Score.

    This works by computing the F1 score per class, and then performs a global weighted average
    according to the inverse of the geometric mean of the class frequency and the total sample size.

    Parameters
    ----------
    cm : ConfusionMatrix
        This parameter allows sharing the same confusion matrix between multiple metrics.
        Sharing a confusion matrix reduces the amount of storage and computation time.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = GeomMeanWeightedF1()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    GeomMeanWeightedF1: 100.00%
    GeomMeanWeightedF1: 31.06%
    GeomMeanWeightedF1: 54.04%
    GeomMeanWeightedF1: 65.53%
    GeomMeanWeightedF1: 62.63%

    """
    
    def __init__(self, cm=None):
        super().__init__(cm)
        
    def update(self, y_true, y_pred):
        self.cm.update(y_true, y_pred)
        return self

    def get(self):
        N = sum(self.cm.sum_row.values())
        weighted_f1_sum = 0
        weight_sum = 0

        for c in self.cm.classes:
            try:
                p = self.cm[c][c] / self.cm.sum_col[c]
            except ZeroDivisionError:
                p = 0

            try:
                r = self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                r = 0

            try:
                f1_c = 2 * p * r / (p + r)
            except ZeroDivisionError:
                f1_c = 0

            # Calculate the geometric mean-based weight for the class
            n_c = self.cm.sum_row[c]
            try:
                w_c = 1 / math.sqrt(n_c * N)
            except ZeroDivisionError:
                w_c = 0

            weighted_f1_sum += w_c * f1_c
            weight_sum += w_c

        return weighted_f1_sum / weight_sum if weight_sum else 0
