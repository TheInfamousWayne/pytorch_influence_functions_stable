from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def from10to2classes(x, y, num_a, num_b):
    is_num_a, is_num_b = y == num_a, y == num_b
    x_2class = np.concatenate([x[is_num_a], x[is_num_b]])
    y_2class = np.concatenate([np.ones(is_num_a.sum()), np.zeros(is_num_b.sum())]).reshape([-1, 1])

    return x_2class, y_2class


def get_2class_mnist(num_a, num_b):
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    x_train, y_train = train_data.data, train_data.targets

    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    x_test, y_test = test_data.data, test_data.targets

    x_train, x_test = x_train.reshape([-1, 784]) / 255.0, x_test.reshape([-1, 784]) / 255.0
    x_train_2class, y_train_2class = from10to2classes(x_train, y_train, num_a, num_b)
    x_test_2class, y_test_2class = from10to2classes(x_test, y_test, num_a, num_b)

    return (x_train_2class, y_train_2class), (x_test_2class, y_test_2class)


def visualize_result(actual_loss_diff, estimated_loss_diff, reverse_loss_diff, reverse_ids):
    r2_s = r2_score(actual_loss_diff, estimated_loss_diff)
    r2_r = r2_score(actual_loss_diff[reverse_ids], reverse_loss_diff)
    mae = mean_absolute_error(actual_loss_diff, estimated_loss_diff)

    max_abs = np.max([np.max(np.abs(actual_loss_diff)),
                      np.max(np.abs(estimated_loss_diff)),
                      np.max(np.abs(reverse_loss_diff))])
    min_, max_ = -max_abs * 1.1, max_abs * 1.1
    plt.rcParams['figure.figsize'] = 6, 5
    plt.scatter(actual_loss_diff, estimated_loss_diff, zorder=2, s=10)
    plt.scatter(actual_loss_diff[reverse_ids], reverse_loss_diff, zorder=2, s=10, c="#ff7f0e")
    plt.title('Loss diff')
    plt.xlabel('Actual loss diff')
    plt.ylabel('Estimated loss diff')
    range_ = [min_, max_]
    plt.plot(range_, range_, 'k-', alpha=0.2, zorder=1)
    text = 'MAE = {:.03}\nR2 score = {:.03}\nR2 reverse = {:.03}'.format(mae, r2_s, r2_r)
    plt.text(max_abs, -max_abs, text, verticalalignment='bottom', horizontalalignment='right')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)

    plt.savefig("result.png")

    return r2_s, r2_r
