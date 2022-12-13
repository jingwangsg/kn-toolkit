from kn_util.config import LazyCall as L


def eval_str_impl(s):
    return eval(s)


def eval_str(s):
    return L(eval_str_impl)(s=s)