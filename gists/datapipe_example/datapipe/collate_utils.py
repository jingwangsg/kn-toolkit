from kn_util.data import fix_tensor_to_float32, merge_list_to_tensor
import warnings
_warn_flag = False

def default_collate_fn(x):
    try:
        return fix_tensor_to_float32(merge_list_to_tensor(x))
    except:
        global _warn_flag
        if not _warn_flag:
            warnings.warn("part of outputs are not tensor")
            _warn_flag = True
        return x