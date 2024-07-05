import unittest
from river.metrics import ConfusionMatrix
from geom_mean_w_f1 import GeomMeanWeightedF1  # Replace with the actual import

class TestGeomMeanWeightedF1(unittest.TestCase):

    def setUp(self):
        self.metric = GeomMeanWeightedF1()
        self.cm = ConfusionMatrix()

    def test_perfect_classification(self):
        y_true = [0, 1, 2, 2, 2]
        y_pred = [0, 1, 2, 2, 2]
        for yt, yp in zip(y_true, y_pred):
            self.metric.update(yt, yp)
        self.assertEqual(self.metric.get(), 1.0)

    def test_imbalanced_classification(self):
        y_true = [0, 0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 0, 0]
        for yt, yp in zip(y_true, y_pred):
            self.metric.update(yt, yp)
        self.assertLess(self.metric.get(), 1.0)

    def test_all_wrong_classification(self):
        y_true = [0, 1, 2]
        y_pred = [2, 0, 1]
        for yt, yp in zip(y_true, y_pred):
            self.metric.update(yt, yp)
        self.assertEqual(self.metric.get(), 0.0)

    def test_empty_classification(self):
        self.assertEqual(self.metric.get(), 0.0)

    def test_single_class(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        for yt, yp in zip(y_true, y_pred):
            self.metric.update(yt, yp)
        self.assertEqual(self.metric.get(), 1.0)

    def test_zero_division(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        for yt, yp in zip(y_true, y_pred):
            self.metric.update(yt, yp)
        self.assertEqual(self.metric.get(), 0.0)

if __name__ == "__main__":
    unittest.main()
