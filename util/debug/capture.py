from ..file import save_pickle
from ..general.logger import get_logger

log = get_logger(__name__)


def capture_to_pickle(output_fn, **kwargs):
    save_pickle(kwargs, output_fn)
    log.info(f"variables captured to {output_fn}")
    import fairseq.pdb as pdb

    pdb.set_trace()  # FIXME

