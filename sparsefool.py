from torch.autograd import Variable
from copy import deepcopy
import torch
import numpy as np
import time
from utils import get_prediction


def deepfool(
    im,
    net,
    lambda_fac=3.0,
    overshoot=0.02,
    step_size=0.01,
    max_iter=50,
    device="cuda",
):
    n_queries = 0
    X0 = deepcopy(im)  # - Keep continuous version
    # assert ((X0 == 0.0) | (X0 == 1.0)).all(), "Non binary input deepfool"

    n_queries += 1
    f_image = net.fw_over_time(Variable(X0, requires_grad=True)).data.cpu().numpy().flatten()
    num_classes = len(f_image)
    input_shape = X0.size()

    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    X_adv = deepcopy(X0)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = label
    loop_i = 0

    while k_i == label and loop_i < max_iter:

        x = Variable(X_adv, requires_grad=True)  # Rounding

        fs = net.fw_over_time(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = deepcopy(x.grad.data)

        for k in range(1, num_classes):
            x.grad.detach_()
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()
            assert not torch.isnan(pert_k).any(), "Found NaN"
            assert not torch.isinf(pert_k).any(), "Found Inf"
            # assert not (cur_grad == 0.0).all(), "Zero cur grad"

            if pert_k < pert:
                pert = pert_k + 0.0
                w = w_k + 0.0

        r_i = torch.clamp(pert, min=step_size) * w / w.norm()  # change here maybe
        assert not torch.isnan(r_i).any(), "Found NaN"
        assert not torch.isinf(r_i).any(), "Found Inf"
        r_tot = r_tot + r_i

        X_adv = X_adv + r_i

        check_fool = X0 + (1 + overshoot) * r_tot

        n_queries += 1
        k_i = torch.argmax(net.fw_over_time(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    x = Variable(X_adv, requires_grad=True)

    n_queries += 1
    fs = net.fw_over_time(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)

    grad = deepcopy(x.grad.data)
    assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"
    if grad.norm() == 0.0:
        print("WARNING Grad norm is zero")
    grad = grad / (grad.norm() + 1e-10)
    assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"

    r_tot = lambda_fac * r_tot
    X_adv = X0 + r_tot

    return grad, X_adv, n_queries


def sparsefool(
    x_0,
    net,
    max_hamming_distance,
    lb=0.0,
    ub=1.0,
    lambda_=2.0,
    max_iter=20,
    epsilon=0.02,
    overshoot=0.02,
    step_size=0.01,
    max_iter_deep_fool=50,
    device="cuda",
    verbose=False,
):
    t0 = time.time()
    n_queries = 1
    x_0 = x_0.to(device)
    pred_label = torch.argmax(net.fw_over_time(Variable(x_0, requires_grad=True)).data).item()

    x_i = deepcopy(x_0)
    fool_im = deepcopy(x_i)

    fool_label = pred_label
    loops = 0

    while fool_label == pred_label and loops < max_iter:

        if verbose:
            print(f"{loops}/{max_iter}")

        normal, x_adv, n_queries_deepfool = deepfool(
            im=x_i,
            net=net,
            lambda_fac=lambda_,
            overshoot=overshoot,
            step_size=step_size,
            max_iter=max_iter_deep_fool,
            device=device,
        )

        assert not torch.isnan(normal).any() and not torch.isnan(x_adv).any(), "Found NaN"
        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im = deepcopy(x_i)
        fool_label = get_prediction(net, fool_im)

        n_queries += n_queries_deepfool + 1
        loops += 1

    X_adv = fool_im
    L0 = int(torch.sum(torch.abs(fool_im - x_0)))

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = (
        1
        if not (pred_label == get_prediction(net, X_adv))
        and L0 <= max_hamming_distance
        else 0
    )
    return_dict["elapsed_time"] = t1 - t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = pred_label
    return_dict["predicted_attacked"] = int(get_prediction(net, X_adv))

    if return_dict["success"]:
        assert pred_label != get_prediction(net, X_adv), "Success but the same label"

    if verbose:
        if return_dict["success"]:
            print("Success, L0 =", L0)
        else:
            print("No success")

    return return_dict


def linear_solver(x_0, normal, boundary_point, lb, ub):
    input_shape = x_0.size()

    coord_vec = deepcopy(normal)
    plane_normal = deepcopy(coord_vec).view(-1)
    plane_point = deepcopy(boundary_point).view(-1)

    x_i = deepcopy(x_0)

    f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
    sign_true = f_k.sign().item()

    beta = 0.001 * sign_true
    current_sign = sign_true

    while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

        pert = f_k.abs() / coord_vec.abs().max()

        mask = torch.zeros_like(coord_vec)
        mask[np.unravel_index(torch.argmax(coord_vec.abs()).item(), input_shape)] = 1.

        r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

        x_i = x_i + r_i
        x_i = torch.clamp(x_i, lb, ub)

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
        current_sign = f_k.sign().item()

        coord_vec[r_i != 0] = 0

    return x_i
