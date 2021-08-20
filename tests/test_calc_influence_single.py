from unittest import TestCase
import torch
import unittest
from sklearn.metrics import r2_score

from pytorch_influence_functions.influence_functions.influence_functions import (
    calc_influence_single,
    calc_img_wise
)

from pytorch_influence_functions.influence_functions.utils import (
    get_default_config
)

from utils.logistic_regression import (
    LogisticRegression,
)


class TestIHVPGrad(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Generate a pytorch model and a dummy dataset:
        cls.n_features = 10
        cls.n_classes = 3
        cls.n_params = cls.n_classes * cls.n_features + cls.n_features
        cls.wd = wd = 0.1
        cls.model = LogisticRegression(cls.n_classes, cls.n_features, wd=cls.wd)
        cls.useless_model = LogisticRegression(cls.n_classes, cls.n_features, wd=cls.wd, add_useless_layer=True)

        # Fill the model with parameters fitting to the dataset (computed beforehand)
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
            cls.useless_model.linear.weight = torch.nn.Parameter(
                torch.tensor(coef, dtype=torch.float)
            )
            cls.useless_model.linear.bias = torch.nn.Parameter(
                torch.tensor(intercept, dtype=torch.float)
            )
            cls.useless_model.useless_layer.weight = torch.nn.Parameter(
                torch.tensor(coef, dtype=torch.float), requires_grad=False
            )
            cls.useless_model.useless_layer.bias = torch.nn.Parameter(
                torch.tensor(intercept, dtype=torch.float), requires_grad=False
            )

        cls.gpu = 1 if torch.cuda.is_available() else -1
        if cls.gpu == 1:
            cls.model = cls.model.cuda()
            cls.useless_model = cls.useless_model.cuda()

    class CreateData(torch.utils.data.Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            out_data = self.data[idx]
            out_label = self.targets[idx]

            return out_data, out_label

    def test_influence_symmetry(self):
        # Tests if it matters whether we measure the influence from a test point on a train point or vice versa.
        # Mathematically, it should deliver the same result since the inverse Hessian is symmetric.
        self.model.eval()

        train_id = [10, 50, 1299, 3920, 1929]
        test_id = [20, 90, 284, 1028, 1828]
        train_loader = torch.utils.data.DataLoader(self.model.training_set, batch_size=1, shuffle=True)

        # Calculate influence from train on test
        train_to_test = []
        for tr_id in train_id:
            (x_target, y_target) = self.model.test_set[test_id]
            target_subset = self.CreateData(x_target, y_target)
            target_loader = torch.utils.data.DataLoader(target_subset, batch_size=1, shuffle=False)
            x_infl, y_infl = self.model.training_set[tr_id]
            x_infl = train_loader.collate_fn([x_infl])
            y_infl = train_loader.collate_fn([y_infl])
            res, _, _ = calc_influence_single(self.model, train_loader, target_loader, x_infl, y_infl, gpu=0,
                                                recursion_depth=500, r=5)
            train_to_test.append(res)
        train_to_test = torch.flatten(torch.tensor(train_to_test))

        # Calculate influence from test on train
        test_to_train = []
        for te_id in test_id:
            (x_target, y_target) = self.model.training_set[train_id]
            target_subset = self.CreateData(x_target, y_target)
            target_loader = torch.utils.data.DataLoader(target_subset, batch_size=1, shuffle=False)
            x_infl, y_infl = self.model.test_set[te_id]
            x_infl = train_loader.collate_fn([x_infl])
            y_infl = train_loader.collate_fn([y_infl])
            res, _, _ = calc_influence_single(self.model, train_loader, target_loader, x_infl, y_infl, gpu=0,
                                                recursion_depth=500, r=5)
            test_to_train.append(res)
        # Transpose so that it gives the same pairs of (x_train, x_test) as train_to_test
        test_to_train = torch.flatten(torch.tensor(test_to_train).t())

        r2 = r2_score(train_to_test, test_to_train)
        print(r2)
        assert r2 > 0.9

    def test_calc_img_wise(self):
        config = get_default_config()
        config["r"] = 1
        config["recursion_depth"] = 1
        train_loader = torch.utils.data.DataLoader(self.model.training_set, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.model.test_set, batch_size=1, shuffle=True)
        calc_img_wise(config, self.model, train_loader, test_loader)

    def test_model_without_full_gradients(self):
        # Test if the influence function is ignoring layers that have requires_grad = True, as it should
        self.model.eval()

        # Generate data
        tr_id = 10
        test_id = [20, 90, 284, 1028, 1828]
        train_loader = torch.utils.data.DataLoader(self.model.training_set, batch_size=1, shuffle=True)
        (x_target, y_target) = self.model.test_set[test_id]
        target_subset = self.CreateData(x_target, y_target)
        target_loader = torch.utils.data.DataLoader(target_subset, batch_size=1, shuffle=False)
        x_infl, y_infl = self.model.training_set[tr_id]
        x_infl = train_loader.collate_fn([x_infl])
        y_infl = train_loader.collate_fn([y_infl])

        torch.manual_seed(20082021)
        resUseless, _, _ = calc_influence_single(self.useless_model, train_loader, target_loader, x_infl, y_infl, gpu=0,
                                                 recursion_depth=500, r=5)
        torch.manual_seed(20082021)
        res, _, _ = calc_influence_single(self.model, train_loader, target_loader, x_infl, y_infl, gpu=0,
                                          recursion_depth=500, r=5)

        r2 = r2_score(res, resUseless)
        print(r2)
        assert r2 >= 0.999

if __name__ == "__main__":
    unittest.main()
