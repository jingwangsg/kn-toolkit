from torch.optim.lr_scheduler import ReduceLROnPlateau
from kn_util.config import LazyCall as L


def reduce_lr_on_plateau(factor=0.9, patience=20, verbose=True):
    return L(ReduceLROnPlateau)(optimizer=None, factor=factor, patience=patience, verbose=verbose)
