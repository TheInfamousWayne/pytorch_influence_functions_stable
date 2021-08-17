import torch
import numpy as np
from torch import nn

from ops import log_clip

class LogisticRegression(torch.nn.Module):
    def __init__(self, weight_decay):
        super().__init__()

        self.wd = torch.FloatTensor([weight_decay]).cuda()
        self.linear = nn.Linear(784, 1)

    def forward(self, x):
        x = self.linear(x)

        return x

    def loss(self, logits, y, train=True):
        preds = torch.sigmoid(logits)

        if train:
            loss = -torch.mean(y * torch.log(preds) + (1 - y) * torch.log(1 - preds)) + torch.sum(self.linear.weight*self.linear.weight) * self.wd / 2
        else:
            loss = -torch.mean(y * torch.log(preds) + (1 - y) * torch.log(1 - preds)) + torch.sum(self.linear.weight*self.linear.weight) * self.wd / 2

        return loss

    def get_inverse_hvp_lissa(self, v, x, y, scale=10, num_samples=5, recursion_depth=1000, print_iter=100):

        inverse_hvp = None

        for i in range(num_samples):
            print('Sample iteration [{}/{}]'.format(i+1, num_samples))
            cur_estimate = v
            permuted_indice = np.random.permutation(range(len(x)))

            for j in range(recursion_depth):

                x_sample = x[permuted_indice[j]:permuted_indice[j]+1]
                y_sample = y[permuted_indice[j]:permuted_indice[j]+1]

                # get hessian vector product
                hvp = self.sess.run(self.hvp, feed_dict={self.x: x_sample,
                                                         self.y: y_sample,
                                                         self.u: cur_estimate})

                # update hv
                cur_estimate = v + cur_estimate - hvp / scale

                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth {}: norm is {}".format(j, np.linalg.norm(cur_estimate)))

            if inverse_hvp is None:
                inverse_hvp = cur_estimate / scale
            else:
                inverse_hvp = inverse_hvp + cur_estimate / scale

        inverse_hvp = inverse_hvp / num_samples
        return inverse_hvp
