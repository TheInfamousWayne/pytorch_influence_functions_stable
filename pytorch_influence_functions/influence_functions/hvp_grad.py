#! /usr/bin/env python3
import torch
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_influence_functions.influence_functions.utils import (
    conjugate_gradient,
    load_weights,
    make_functional,
    tensor_to_tuple,
    display_progress
)


def s_test_cg(x_test, y_test, model, train_loader, damp, gpu=-1, verbose=True):

    if gpu >= 0:
        x_test, y_test = x_test.cuda(), y_test.cuda()

    v_flat = parameters_to_vector(grad_z(x_test, y_test, model, gpu))

    def hvp_fn(x):

        x_tensor = torch.tensor(x, requires_grad=False)
        if gpu >= 0:
            x_tensor = x_tensor.cuda()

        params, names = make_functional(model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        hvp = torch.zeros_like(flat_params)

        for x_train, y_train in train_loader:

            if gpu >= 0:
                x_train, y_train = x_train.cuda(), y_train.cuda()

            def f(flat_params_):
                split_params = tensor_to_tuple(flat_params_, params)
                load_weights(model, names, split_params)
                out = model(x_train)
                loss = model.loss(out, y_train)
                return loss

            batch_hvp = vhp(f, flat_params, x_tensor, strict=True)[1]

            hvp += batch_hvp / float(len(train_loader))

        with torch.no_grad():
            load_weights(model, names, params, as_params=True)
            damped_hvp = hvp + damp * v_flat

        return damped_hvp.cpu().numpy()

    def print_function_value(_, f_linear, f_quadratic):
        print(
            f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
        )

    debug_callback = print_function_value if verbose else None

    result = conjugate_gradient(
        hvp_fn,
        v_flat.cpu().numpy(),
        debug_callback=debug_callback,
        avextol=1e-8,
        maxiter=100,
    )

    result = torch.tensor(result)
    if gpu >= 0:
        result = result.cuda()

    return result


def s_test(x_test, y_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0, recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        z_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor

    Returns:
        h_estimate: list of torch tensors, s_test"""

    # v = grad_z(x_test, y_test, model, gpu)
    # h_estimate = v
    #
    # params, names = make_functional(model)
    # req_grad = [p.requires_grad for p in params]
    # # Make params regular Tensors instead of nn.Parameter and do not lose gradient information along the way
    # params = tuple(p.detach().requires_grad_() if r else p.detach() for (r, p) in zip(req_grad, params))
    #
    # # Only calculate s_test with respect to parameters that require grad:
    # params_grad = tuple(p for (r, p) in zip(req_grad, params) if r)
    # names_grad = [n for (r, n) in zip(req_grad, names) if r]
    # params_nograd = tuple(p for (r, p) in zip(req_grad, params) if not r)
    # names_nograd = [n for (r, n) in zip(req_grad, names) if not r]
    #
    # # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    # progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    # for i, (x_train, y_train) in enumerate(progress_bar):
    #
    #     if gpu >= 0:
    #         x_train, y_train = x_train.cuda(), y_train.cuda()
    #
    #     def f(*new_params):
    #         load_weights(model, names_grad, new_params)
    #         load_weights(model, names_nograd, params_nograd)
    #         out = model(x_train)
    #         loss = model.loss(out, y_train)
    #         return loss
    #
    #     hv = vhp(f, params_grad, tuple(h_estimate), strict=True)[1]
    #
    #     # Recursively calculate h_estimate
    #     with torch.no_grad():
    #         h_estimate = [
    #             _v + (1 - damp) * _h_e - _hv / scale
    #             for _v, _h_e, _hv in zip(v, h_estimate, hv)
    #         ]
    #
    #         if i % 100 == 0:
    #             norm = sum([h_.norm() for h_ in h_estimate])
    #             progress_bar.set_postfix({"est_norm": norm.item()})
    #
    # with torch.no_grad():
    #     load_weights(model, names, params, as_params=True)
    #     for (r, p) in zip(req_grad, model.parameters()):
    #         p.requires_grad_(r)
    #
    # return h_estimate

    v = grad_z(x_test, y_test, model, gpu)
    h_init_estimates = v

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            y = model(x)
            loss = model.loss(y, t)
            hv = hvp(loss, list(model.parameters()), h_init_estimates)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * h_estimate - _hv / scale
                for _v, h_estimate, _hv in zip(v, h_init_estimates, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def grad_z(x, y, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()

    # initialize
    if gpu >= 0:
        x, y = x.cuda(), y.cuda()

    prediction = model(x)

    loss = model.loss(prediction, y)

    # Compute sum of gradients from model parameters to loss
    # for all model parameters that require gradients
    params = [param for param in model.parameters() if param.requires_grad]
    return grad(loss, params)


def s_test_sample(
    model,
    x_test,
    y_test,
    train_loader,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    # inverse_hvp = [
    #     torch.zeros_like(params, dtype=torch.float) for params in model.parameters() if params.requires_grad
    # ]
    #
    # for i in range(r):
    #
    #     hessian_loader = DataLoader(
    #         train_loader.dataset,
    #         sampler=torch.utils.data.RandomSampler(
    #             train_loader.dataset, True, num_samples=recursion_depth * train_loader.batch_size
    #         ),
    #         batch_size=train_loader.batch_size,
    #         num_workers=4,
    #     )
    #
    #     cur_estimate = s_test(
    #         x_test, y_test, model, i, hessian_loader, gpu=gpu, damp=damp, scale=scale
    #     )
    #
    #     with torch.no_grad():
    #         inverse_hvp = [
    #             old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate)
    #         ]
    #
    # with torch.no_grad():
    #     inverse_hvp = [component / r for component in inverse_hvp]
    #
    # return inverse_hvp

    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(x_test, y_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        display_progress("Averaging r-times: ", i, r)

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads