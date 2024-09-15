import gc
import inspect
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from termcolor import colored
from torch import Tensor


# https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/13
def collect_tensor_infos_gc(only_cuda=False, omit_objs=[]):
    """

    :return: list of active PyTorch tensors
    >>> import torch
    >>> from torch import tensor
    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1], device=device)
    >>> a3 = tensor([[1, 2], [3, 4]], device=device)
    >>> # print(get_all_tensor_names())
    >>> tensors = [tensor_obj for tensor_obj in get_tensors(only_cuda=only_cuda)]
    >>> # print(tensors)
    >>> # We doubled each t1, a3 tensors because of the tensors collection.
    >>> expected_tensor_length = 2
    >>> assert len(tensors) == expected_tensor_length, f"Expected length of tensors {expected_tensor_length}, but got {len(tensors)}, the tensors: {tensors}"
    >>> exp_size = (2,2)
    >>> act_size = tensors[1].size()
    >>> assert exp_size == act_size, f"Expected size {exp_size} but got: {act_size}"
    >>> del t1
    >>> del a3
    >>> clean_gc_return = map((lambda obj: del_object(obj)), tensors)
    """

    add_all_tensors = False if only_cuda is True else True
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensor_infos = dict()

    # omit_obj_ids = [id(obj) for obj in omit_objs]

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, "data") and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensor_infos[id(tensor)] = encode_tensor_info(tensor)

    gc.collect()

    for obj in gc.get_objects():
        if sys.getrefcount(obj) == 0:
            # if no references to the object, it will be garbage collected later
            continue
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, "saved_tensors") and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(obj, "grad") and (id(obj.grad) not in omit_objs):
                    add_tensor(obj.grad)
        except Exception:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensor_infos


# def is_variable(obj, name):
#     return name.startswith("__") or inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isfunction(obj)


# def collect_variable_names():
#     context_dict = dict()
#     frame_infos = inspect.stack()

#     for frame_info in frame_infos[1:]:
#         frame = frame_info.frame
#         local_dict = frame.f_locals
#         global_dict = frame.f_globals
#         module_name = global_dict["__name__"]
#         function_name = module_name + "." + frame_info.function + "()"

#         for name, obj in local_dict.items():
#             if is_variable(obj, name):
#                 continue
#             if name in context_dict:
#                 continue
#             _name = f"{function_name}:{name}"
#             context_dict[_name] = obj

#         for name, obj in global_dict.items():
#             if is_variable(obj, name):
#                 continue
#             if name in context_dict:
#                 continue
#             _name = f"{module_name}:{name}"
#             context_dict[_name] = obj

#     return context_dict


# def collect_named_tensors_gc(include_globals=True, only_cuda=False):
#     """
#     We
#     """
#     # collect all objects from all frames
#     context_dict = collect_variable_names()

#     gc.collect()
#     # create mapping of object id to object name
#     id2name = dict()
#     for name, obj in context_dict.items():

#         def _register_tensor(name, _obj):
#             if id(_obj) in id2name:
#                 return

#             id2name[id(_obj)] = (name, type(_obj))
#             if hasattr(_obj, "saved_tensors") and _obj.saved_tensors is not None:
#                 for idx, tensor in enumerate(_obj.saved_tensors):
#                     _sub_name = f"{name}.saved_tensors[{idx}]"
#                     id2name[id(tensor)] = _sub_name

#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 if hasattr(_obj, "grad") and _obj.grad is not None:
#                     _sub_name = f"{name}.grad"
#                     id2name[id(_obj.grad)] = _sub_name

#         if isinstance(obj, nn.Module):
#             for sub_name, sub_obj in obj.named_parameters():
#                 _sub_name = f"{name}.{sub_name}"
#                 _register_tensor(_sub_name, sub_obj)
#             for sub_name, sub_obj in obj.named_buffers():
#                 _sub_name = f"{name}.{sub_name}"
#                 _register_tensor(_sub_name, sub_obj)

#         if isinstance(obj, torch.optim.Optimizer):
#             for idx, group in obj.state_dict()["state"].items():
#                 for k in group.keys():
#                     if torch.is_tensor(group[k]):
#                         _name = f"{name}.state[{idx}].{k}"
#                         _register_tensor(_name, group[k])

#         if torch.is_tensor(obj):
#             _register_tensor(name, obj)

#     tensor_dict = collect_active_tensors_gc(only_cuda=only_cuda)
#     named_tensor_dict = dict()
#     unresolved_tensors = dict()
#     for name, tensor in tensor_dict.items():
#         if id(tensor) in id2name:
#             named_tensor_dict[id2name[id(tensor)][0]] = tensor
#         else:
#             unresolved_tensors[id(tensor)] = tensor

#     print(colored(f"Named: {len(named_tensor_dict)} Unresolved: {len(unresolved_tensors)}", "red"))

#     return named_tensor_dict, unresolved_tensors


id2context = dict()


def collect_tensor_contexts_gc(desc=None, only_cuda=False):
    tensor_infos = collect_tensor_infos_gc(only_cuda=only_cuda)
    ret_contexts = dict()
    unresolved = dict()

    for tensor_id, tensor_info in tensor_infos.items():
        if tensor_id in id2context:
            # can be traced using register
            ret_contexts[tensor_id] = {"context": id2context[tensor_id], **tensor_info}
        else:
            unresolved[tensor_id] = tensor_info

    if desc:
        print(colored(desc, "red"))
    print(colored(f"Named: {len(ret_contexts)} Unresolved: {len(unresolved)}", "red"))

    return ret_contexts, unresolved


def encode_tensor_info(tensor):
    info = dict(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=str(tensor.device),
    )
    if torch.is_floating_point(tensor):
        info["norm"] = tensor.norm().item()
    elif torch.is_complex(tensor):
        info["norm"] = torch.abs(tensor).norm().item()
    elif tensor.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        info["min"] = tensor.min().item()
        info["max"] = tensor.max().item()
    elif tensor.dtype == torch.bool():
        info["true"] = tensor.sum().item()
        info["false"] = (1 - tensor).sum().item()

    return info


# refer to https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llama_flash_attn_monkey_patch.py
def patch_tensor_with_register():
    """
    By patching all possible interfaces that can create tensors,
    we can register the tensor with the context wherever it is created.
    For each tensor, we store the context (not object reference, we don't want to affect refcount)
    NOTE: to use this function, torch script will be disabled,
    most of the functions below are not compatible to jit

    """

    # have to disable torch scrip here
    def null_script(
        obj,
        optimize=None,
        _frames_up=0,
        _rcb=None,
        example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
    ):
        return obj

    torch.jit.script = null_script

    def _register(tensor):
        frame_infos = inspect.stack()

        recent_locals = frame_infos[1].frame.f_locals
        funcname = None
        if "func" in recent_locals:
            funcname = recent_locals["func"]
        elif "old_arange" in recent_locals:
            funcname = "arange"
        elif "old_init" in recent_locals:
            funcname = "__init__"
        else:
            raise NotImplementedError(f"Unsupported function: {recent_locals}")

        frame_infos = frame_infos[2:]

        def _register_tensor(tensor):
            context_item = [
                dict(
                    code_context=frame_info.code_context,
                    function_name=frame_info.function,
                    code_loc=f"{frame_info.filename}:{frame_info.lineno}",
                )
                for frame_info in frame_infos
            ]
            context_item = [{"func": funcname}] + context_item

            # info = context_item
            id2context[id(tensor)] = context_item

        if torch.is_tensor(tensor):
            _register_tensor(tensor)
        elif isinstance(tensor, List):
            for t in tensor:
                _register_tensor(t)
        elif isinstance(tensor, Tuple):
            for t in tensor:
                _register_tensor(t)
        else:
            warnings.warn(f"Unsupported tensor type: {type(tensor)}")
            return

    old_init = torch.Tensor.__init__

    def new_init(self, *args: Any, **kwargs):
        old_init(self)  # seems like __new__ is responsible for dealing with *args and **kwargs
        _register(self)

    torch.Tensor.__init__ = new_init

    def new_func_builder(func):
        def new_func(*args, **kwargs):
            tensor = func(*args, **kwargs)
            _register(tensor)
            return tensor

        return new_func

    torch_funcs = [
        "mkldnn_convolution",
        "bitwise_xor",
        "quantized_max_pool2d",
        "fused_moving_avg_obs_fake_quant",
        "bilinear",
        "roll",
        "feature_dropout",
        "repeat_interleave",
        "diagonal_scatter",
        "blackman_window",
        "segment_reduce",
        "concatenate",
        "miopen_convolution_add_relu",
        "hypot",
        "addmv",
        "det",
        "eq",
        "sin",
        "triu",
        "linspace",
        "normal",
        "frobenius_norm",
        "gather",
        "sign",
        "binomial",
        "ravel",
        "take",
        "ldexp",
        "le",
        "empty_permuted",
        "binary_cross_entropy_with_logits",
        "as_strided",
        "clamp_min",
        "quantized_max_pool1d",
        "arccosh",
        "argmax",
        "digamma",
        "randperm",
        "log1p",
        "resolve_neg",
        "argmin",
        "fbgemm_linear_int8_weight_fp32_activation",
        "max",
        "heaviside",
        "bernoulli",
        "conj_physical",
        "trunc",
        "kl_div",
        "atan",
        "expm1",
        "tril",
        "tril_indices",
        "round",
        "polar",
        "outer",
        "vsplit",
        "clamp_max",
        "bucketize",
        "cudnn_convolution_transpose",
        "multiply",
        "sparse_compressed_tensor",
        "cholesky",
        "addr",
        "gradient",
        "asin",
        "fbgemm_pack_gemm_matrix_fp16",
        "subtract",
        "bartlett_window",
        "int_repr",
        "arcsin",
        "max_pool2d",
        "ones_like",
        "t_copy",
        "batch_norm_elemt",
        "neg",
        "cosh",
        "addcdiv",
        "erfc",
        "index_put",
        "tensor_split",
        "sspaddmm",
        "isin",
        "miopen_depthwise_convolution",
        "quantized_rnn_relu_cell",
        "ormqr",
        "cross",
        "sub",
        "quantized_rnn_tanh_cell",
        "rand",
        "hsplit",
        "conv1d",
        "sparse_bsr_tensor",
        "hann_window",
        "bincount",
        "greater",
        "put",
        "conv3d",
        "selu",
        "vander",
        "full",
        "sqrt",
        "empty_quantized",
        "grid_sampler",
        "conj",
        "atan2",
        "empty",
        "zeros",
        "nonzero_static",
        "nextafter",
        "isfinite",
        "as_strided_copy",
        "maximum",
        "stack",
        "mm",
        "frac",
        "randint_like",
        "polygamma",
        "relu",
        "range",
        "fmax",
        "cudnn_grid_sampler",
        "norm_except_dim",
        "group_norm",
        "median",
        "cos",
        "floor",
        "cudnn_convolution",
        "embedding",
        "exp2",
        "conv_tbc",
        "diagonal_copy",
        "sparse_csc_tensor",
        "alias_copy",
        "index_fill",
        "sum",
        "diagonal",
        "bmm",
        "inverse",
        "index_copy",
        "lcm",
        "smm",
        "arctan",
        "conv_transpose1d",
        "logical_and",
        "isposinf",
        "sparse_csr_tensor",
        "avg_pool1d",
        "abs",
        "ccol_indices_copy",
        "unsafe_chunk",
        "col_indices_copy",
        "gt",
        "greater_equal",
        "mkldnn_max_pool2d",
        "kaiser_window",
        "multinomial",
        "cudnn_convolution_add_relu",
        "diff",
        "masked_select",
        "pairwise_distance",
        "nansum",
        "xlogy",
        "masked_scatter",
        "imag",
        "narrow",
        "adaptive_avg_pool1d",
        "tensor",
        "sigmoid",
        "unflatten",
        "isneginf",
        "dsplit",
        "clone",
        "remainder",
        "arccos",
        "addmm",
        "not_equal",
        "quantized_max_pool3d",
        "unbind",
        "cosine_similarity",
        "deg2rad",
        "dropout",
        "detach_copy",
        "max_pool1d",
        "q_per_channel_zero_points",
        "chunk",
        "q_per_channel_scales",
        "masked_fill",
        "istft",
        "unfold_copy",
        "unsqueeze_copy",
        "mvlgamma",
        "empty_strided",
        "hardshrink",
        "resolve_conj",
        "feature_alpha_dropout",
        "minimum",
        "transpose",
        "lu_solve",
        "swapdims",
        "atanh",
        "logical_xor",
        "argsort",
        "addbmm",
        "column_stack",
        "orgqr",
        "miopen_convolution_relu",
        "split_with_sizes",
        "negative",
        "fmin",
        "triplet_margin_loss",
        "floor_divide",
        "signbit",
        "kron",
        "argwhere",
        "msort",
        "bitwise_and",
        "log_softmax",
        "pdist",
        "celu",
        "logsumexp",
        "mean",
        "layer_norm",
        "select_scatter",
        "dist",
        "tile",
        "vstack",
        "dot",
        "rrelu",
        "adjoint",
        "sgn",
        "rot90",
        "poisson",
        "log10",
        "sparse_bsc_tensor",
        "trace",
        "combinations",
        "cumprod",
        "cudnn_convolution_relu",
        "arctan2",
        "igammac",
        "amin",
        "copysign",
        "index_reduce",
        "mkldnn_adaptive_avg_pool2d",
        "squeeze",
        "as_tensor",
        "less_equal",
        "dsmm",
        "alpha_dropout",
        "randn",
        "permute",
        "float_power",
        "poisson_nll_loss",
        "corrcoef",
        "asinh",
        "broadcast_to",
        "fix",
        "grid_sampler_3d",
        "scatter_reduce",
        "rsub",
        "logical_not",
        "pow",
        "matrix_power",
        "arcsinh",
        "hspmm",
        "searchsorted",
        "logical_or",
        "native_channel_shuffle",
        "triu_indices",
        "baddbmm",
        "spmm",
        "as_strided_scatter",
        "index_add",
        "expand_copy",
        "batch_norm",
        "affine_grid_generator",
        "sparse_coo_tensor",
        "logaddexp2",
        "quantize_per_tensor_dynamic",
        "gcd",
        "mul",
        "acosh",
        "eye",
        "from_numpy",
        "ge",
        "rad2deg",
        "erfinv",
        "fbgemm_linear_fp16_weight_fp32_activation",
        "diag",
        "movedim",
        "concat",
        "lt",
        "empty_like",
        "prod",
        "reciprocal",
        "log",
        "fbgemm_pack_quantized_matrix",
        "cudnn_affine_grid_generator",
        "cov",
        "positive",
        "miopen_convolution",
        "scalar_tensor",
        "select",
        "renorm",
        "isreal",
        "max_pool3d",
        "fbgemm_linear_fp16_weight",
        "rand_like",
        "true_divide",
        "asarray",
        "fmod",
        "cat",
        "exp",
        "instance_norm",
        "cumsum",
        "hstack",
        "tan",
        "frombuffer",
        "acos",
        "diag_embed",
        "flatten",
        "bitwise_or",
        "var",
        "div",
        "scatter",
        "fbgemm_linear_int8_weight",
        "bitwise_not",
        "matmul",
        "miopen_convolution_transpose",
        "nuclear_norm",
        "row_indices_copy",
        "logdet",
        "pinverse",
        "constant_pad_nd",
        "count_nonzero",
        "conv_transpose3d",
        "t",
        "erf",
        "bitwise_left_shift",
        "i0",
        "fake_quantize_per_tensor_affine",
        "arange",
        "indices_copy",
        "ones",
        "rnn_relu_cell",
        "log2",
        "slice_copy",
        "complex",
        "fliplr",
        "pixel_unshuffle",
        "permute_copy",
        "isinf",
        "batch_norm_backward_elemt",
        "igamma",
        "conv2d",
        "channel_shuffle",
        "unsqueeze",
        "grid_sampler_2d",
        "pixel_shuffle",
        "dstack",
        "clip",
        "convolution",
        "sinh",
        "nanmean",
        "crow_indices_copy",
        "amax",
        "view_as_complex",
        "unsafe_split",
        "dequantize",
        "lerp",
        "fake_quantize_per_channel_affine",
        "absolute",
        "full_like",
        "hamming_window",
        "histc",
        "cosine_embedding_loss",
        "ger",
        "ne",
        "moveaxis",
        "select_copy",
        "matrix_exp",
        "nanmedian",
        "view_as_real",
        "less",
        "quantized_gru_cell",
        "clamp",
        "quantized_batch_norm",
        "gru_cell",
        "softmax",
        "nan_to_num",
        "bitwise_right_shift",
        "hinge_embedding_loss",
        "from_file",
        "addcmul",
        "isclose",
        "sinc",
        "take_along_dim",
        "ceil",
        "quantize_per_tensor",
        "swapaxes",
        "threshold",
        "flipud",
        "trapz",
        "tanh",
        "randn_like",
        "lgamma",
        "detach",
        "margin_ranking_loss",
        "logcumsumexp",
        "native_norm",
        "add",
        "prelu",
        "arctanh",
        "scatter_add",
        "logaddexp",
        "mkldnn_max_pool3d",
        "values_copy",
        "cholesky_solve",
        "ctc_loss",
        "logspace",
        "quantize_per_channel",
        "nanquantile",
        "square",
        "logit",
        "randint",
        "inner",
        "squeeze_copy",
        "min",
        "zeros_like",
        "nonzero",
        "std",
        "unsafe_split_with_sizes",
        "view_as_real_copy",
        "mv",
        "diagflat",
        "conv_transpose2d",
        "saddmm",
        "narrow_copy",
        "cholesky_inverse",
        "fill",
        "slice_scatter",
        "row_stack",
        "angle",
        "divide",
        "hsmm",
        "vdot",
        "isnan",
        "real",
        "rnn_tanh_cell",
        "rsqrt",
        "quantile",
        "trapezoid",
        "index_select",
        "view_copy",
        "flip",
        "reshape",
        "transpose_copy",
        "view_as_complex_copy",
        "cumulative_trapezoid",
        "where",
    ]

    for func_name in torch_funcs:
        old_func = getattr(torch, func_name)
        setattr(torch, func_name, new_func_builder(old_func))


    tensor_funcs = [
        "__abs__",
        "__add__",
        "__and__",
        "__bool__",
        "__complex__",
        "__div__",
        "__eq__",
        "__float__",
        "__floordiv__",
        "__ge__",
        "__getitem__",
        "__gt__",
        "__iadd__",
        "__iand__",
        "__idiv__",
        "__ifloordiv__",
        "__ilshift__",
        "__imod__",
        "__imul__",
        "__index__",
        "__int__",
        "__invert__",
        "__ior__",
        "__irshift__",
        "__isub__",
        "__ixor__",
        "__le__",
        "__long__",
        "__lshift__",
        "__lt__",
        "__matmul__",
        "__mod__",
        "__mul__",
        "__ne__",
        "__neg__",
        "__new__",
        "__nonzero__",
        "__or__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rfloordiv__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__sub__",
        "__truediv__",
        "__xor__",
        "argmax",
        "argmin",
        "clone",
        "detach",
        "exp",
        "expand",
        "log",
        "max",
        "mean",
        "min",
        "pow",
        "relu",
        "sigmoid",
        "softmax",
        "sqrt",
        "sum",
        "tanh",
        "topk",
    ]
    torch.Tensor.sum
    for func_name in tensor_funcs:
        old_func = getattr(torch.Tensor, func_name)
        setattr(torch.Tensor, func_name, new_func_builder(old_func))

    torch.Tensor.tanh

    # old_tensor = torch.tensor

    # def new_tensor(
    #     data: Any,
    #     dtype: Optional[_dtype] = None,
    #     device: Device = None,
    #     requires_grad: _bool = False,
    # ):
    #     ret = old_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.tensor = new_tensor

    # old_ones = torch.ones

    # def new_ones(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_ones(*size, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.ones = new_ones

    # old_detach = torch.Tensor.detach

    # def new_detach(self):
    #     ret = old_detach(self)
    #     _register(ret)
    #     return ret

    # torch.Tensor.detach = new_detach

    # old_empty = torch.empty

    # def new_empty(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_empty(*size, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.empty = new_empty

    # old_eye = torch.eye

    # def new_eye(
    #     n,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    # ):
    #     ret = old_eye(n, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.eye = new_eye

    # old_zeros = torch.zeros

    # def new_zeros(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_zeros(*size, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.zeros = new_zeros

    # old_clone = torch.Tensor.clone

    # def new_clone(self, memory_format: Optional[_layout] = None):
    #     ret = old_clone(self, memory_format=memory_format)
    #     _register(ret)
    #     return ret

    # torch.clone = new_clone

    # old_full = torch.full

    # def new_full(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_full(size, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.full = new_full

    # old_ones_like = torch.ones_like

    # def new_ones_like(
    #     input: Tensor,
    #     memory_format: Optional[memory_format] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     pin_memory: Optional[_bool] = False,
    #     requires_grad: Optional[_bool] = False,
    # ):
    #     ret = old_ones_like(
    #         input,
    #         memory_format=memory_format,
    #         dtype=dtype,
    #         layout=layout,
    #         device=device,
    #         pin_memory=pin_memory,
    #         requires_grad=requires_grad,
    #     )
    #     _register(ret)
    #     return ret

    # torch.ones_like = new_ones_like

    # old_rand = torch.rand

    # def new_rand(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_rand(size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.rand = new_rand

    # old_randn = torch.randn

    # def new_randn(
    #     *size,
    #     out: Optional[Tensor] = None,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    #     **kwargs,
    # ):
    #     ret = old_randn(size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.randn = new_randn

    # old_zeros_like = torch.zeros_like

    # def new_zeros_like(
    #     input: Tensor,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    # ):
    #     ret = old_zeros_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.zeros_like = new_zeros_like

    # old_rand_like = torch.rand_like

    # def new_rand_like(
    #     input: Tensor,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    # ):
    #     ret = old_rand_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.rand_like = new_rand_like

    # old_randn_like = torch.randn_like

    # def new_randn_like(
    #     input: Tensor,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    # ):
    #     ret = old_randn_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    #     _register(ret)
    #     return ret

    # torch.randn_like = new_randn_like

    # old_empty_like = torch.empty_like

    # def new_empty_like(
    #     input: Tensor,
    #     dtype: Optional[_dtype] = None,
    #     layout: Optional[_layout] = None,
    #     device: Optional[Union[_device, str, None]] = None,
    #     requires_grad: Optional[_bool] = False,
    #     memory_format: Optional[memory_format] = None,
    # ):
    #     ret = old_empty_like(
    #         input,
    #         dtype=dtype,
    #         layout=layout,
    #         device=device,
    #         requires_grad=requires_grad,
    #         memory_format=memory_format,
    #     )
    #     _register(ret)
    #     return ret

    # torch.empty_like = new_empty_like

    old_arange = torch.arange

    def new_arange(
        *args,
        step=1,
        out: Optional[Tensor] = None,
        dtype: Optional[_dtype] = None,
        layout: Optional[_layout] = None,
        device: Optional[Union[_device, str, None]] = None,
        requires_grad: Optional[_bool] = False,
        **kwargs,
    ):
        if len(args) == 1:
            start, end = 0, args[0]
        elif len(args) == 2:
            start, end = args
        else:
            start, end, step = args

        ret = old_arange(start, end, step=step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        _register(ret)
        return ret

    torch.arange = new_arange
