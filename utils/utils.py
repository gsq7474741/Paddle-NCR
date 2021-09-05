# from x2paddle import torch2paddle
import logging
import warnings
from collections import defaultdict

import numpy as np
import paddle

from configs import cfg

LOWER_METRIC_LIST = ['rmse', 'mae']

TYPE_ORDER = ["bool", "int32", "int64", "float16", "float32", "float64"]
TYPE_MAPPER = {"fp16": "float16", "fp32": "float32", "fp64": "float64"}


def concat(tensors, dim=0):
    x = tensors
    last_index = -1
    for ele in x:
        t = str(ele.dtype).lower().strip().split(".")[-1]
        if t in TYPE_MAPPER:
            t = TYPE_MAPPER[t]
        index = TYPE_ORDER.index(t)
        if last_index < index:
            last_index = index
    real_type = TYPE_ORDER[last_index]
    x = list(x)
    for i in range(len(x)):
        x[i] = x[i].cast(real_type)
    return paddle.concat(x, dim)


def update_parameters(parameters, lr, weight_decay):
    parameters_list = list()
    if parameters is not None:
        for items in parameters:
            if isinstance(items, dict):
                params = items["params"]
                if "lr" in items:
                    for p in params:
                        p.optimize_attr["learning_rate"] = items[
                                                               "lr"] / lr * p.optimize_attr["learning_rate"]
                if "weight_decay" in items:
                    for p in params:
                        if isinstance(items["weight_decay"], (float, int)):
                            p.regularizer = paddle.regularizer.L2Decay(items["weight_decay"])
                        else:
                            p.regularizer = weight_decay
                for p in params:
                    print(p.regularizer)
                parameters_list.extend(params)
            else:
                parameters_list.append(items)
    return parameters_list


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Momentum(paddle.optimizer.Momentum):
    def __init__(self,
                 params,
                 lr=0.001,
                 momentum=0.0,
                 dampening=0,
                 weight_decay=0.0,
                 nesterov=False):
        assert dampening == 0, "The dampening must be 0 in Momentum!"
        parameters_list = update_parameters(params, lr, weight_decay)
        super().__init__(
            learning_rate=lr,
            momentum=momentum,
            parameters=parameters_list,
            use_nesterov=nesterov,
            weight_decay=weight_decay,
            grad_clip=None,
            name=None)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov)
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors.")
            # if not param.is_leaf:
            #     raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; ",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        return self.clear_grad()


class Adam(paddle.optimizer.Adam):
    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 amsgrad=False):
        parameters_list = update_parameters(params, lr, weight_decay)
        if weight_decay == 0:
            weight_decay = None
        super().__init__(
            learning_rate=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=eps,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=None,
            name=None,
            lazy_mode=False)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(parameters_list)
        if len(param_groups) == 0:
            print(param_groups)
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors.")
            # if not param.is_leaf:
            #     raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; ",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        return self.clear_grad()


def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default=cfg.DEFAULT_LOG, help='Logging file path')
    parser.add_argument('--result_file', type=str, default=cfg.DEFAULT_RESULT, help='Result file path')
    parser.add_argument('--random_seed', type=int, default=2022, help='Random seed of numpy and pytorch')
    parser.add_argument('--train', type=int, default=1, help='To train the saved_model or not.')
    return parser


def balance_data(data):
    pos_indexes = np.where(data['Y'] == 1)[0]
    copy_num = int((len(data['Y']) - len(pos_indexes)) / len(pos_indexes))
    if copy_num > 1:
        copy_indexes = np.tile(pos_indexes, copy_num)
        sample_index = np.concatenate([np.arange(0, len(data['Y'])),
                                       copy_indexes])
        for k in data:
            data[k] = data[k][sample_index]
    return data


def input_data_is_list(data):
    if type(data) is list or type(data) is tuple:
        print('input_data_is_list')
        new_data = {}
        for k in data[0]:
            new_data[k] = np.concatenate([d[k] for d in data])
        return new_data
    return data


def format_metric(metric):
    """
    convert output into string
    :param metric:
    :return:
    """
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m
                                                               ) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m
                                                             ) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def shuffle_in_unison_scary(data):
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def strictly_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    return non_increasing(l) or non_decreasing(l)


def numpy_to_torch(d):
    t = paddle.to_tensor(d)
    if paddle.device.get_device() != 'cpu':
        t = t.cuda()
    return t
