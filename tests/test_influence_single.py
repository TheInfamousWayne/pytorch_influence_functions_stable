from unittest import TestCase
import torch
import pytorch_lightning as pl
import unittest

from pytorch_influence_functions.influence_functions.hvp_grad import (
    grad_z,
)

from utils.logistic_regression import (
    LogisticRegression,
)


class TestIHVPGrad(TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        cls.n_features = 10
        cls.n_classes = 3

        cls.n_params = cls.n_classes * cls.n_features + cls.n_features

        # Use l2 regularization since SklearnLogReg down below uses it, too
        # (loss functions need to be aligned)
        cls.wd = wd = 1
        cls.model = LogisticRegression(cls.n_classes, cls.n_features, wd=cls.wd)

        coef = [[-0.114, -0.251, 0.103, -0.299, -0.294, 0.010, -0.182, 0.474, -0.688, -0.414],
                [0.283, 0.149, -0.064, -0.052, 0.363, 0.048, -0.183, -0.337, 0.277, -0.086],
                [-0.169, 0.102, -0.039, 0.351, -0.069, -0.058, 0.365, -0.136, 0.411, 0.501]]
        intercept = [0.105, 0.197, -0.302]

        with torch.no_grad():
            cls.model.linear.weight = torch.nn.Parameter(
                torch.tensor(coef, dtype=torch.float)
            )
            cls.model.linear.bias = torch.nn.Parameter(
                torch.tensor(intercept, dtype=torch.float)
            )

        cls.gpu = 1 if torch.cuda.is_available() else -1

        if cls.gpu == 1:
            cls.model = cls.model.cuda()

    def test_z_grad_single(self):
        self.model.eval()
        # Setup test point data
        test_idx = 5
        x_test = torch.tensor(
            self.model.test_set.data[[test_idx]], dtype=torch.float
        )
        y_test = torch.tensor(
            self.model.test_set.targets[[test_idx]], dtype=torch.long
        )

        train_loader = self.model.train_dataloader(batch_size=1, shuffle=False)
        x, y = next(iter(train_loader))
        # Compute anc flatten grad
        grads = grad_z(x, y, self.model, gpu=self.gpu)

        print(grads)


if __name__ == "__main__":
    unittest.main()
