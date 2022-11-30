import copy
from kn_util.debug import dict_diff
from kn_util.general import registry, get_logger
import time
from pprint import pformat
from tqdm import tqdm

log = get_logger(__name__)


def processor_profiler(fn):
    def wrapper(**kwargs):
        batch = kwargs["batch"]
        processor = kwargs["processor"]
        _signal = "_TEST_PIPELINE_SIGNAL"
        elem_copy = copy.copy(batch[0])
        _st = time.time()
        ret = fn(**kwargs)
        verbose = registry.get_object(_signal, False)
        if verbose:
            log.info(
                f"\napply [processor] {type(processor).__name__} (costs {time.time() - _st:3f} s)\n"
                + dict_diff(elem_copy, batch[0])
            )
        return ret

    return wrapper


@processor_profiler
def apply_single_processor(batch, processor, tqdm_args=None):
    if getattr(processor, "is_batch_processor", False):
        batch = processor(batch)
    else:
        if tqdm_args:
            for idx, e in tqdm(enumerate(batch), **tqdm_args):
                batch[idx] = processor(e)
        else:
            for idx, e in enumerate(batch):
                batch[idx] = processor(e)
    return batch


def apply_processors(batch, processors, tqdm_args=None):
    for processor in processors:
        batch = apply_single_processor(batch=batch, processor=processor, tqdm_args=tqdm_args)
    return batch


def build_processors(processors_cfg, verbose=False):
    processors = []
    for processor_cfg in processors_cfg:
        processors += [registry.build_from_cfg(processor_cfg, "processor")]
    if verbose:
        log.info("\n===============processors built==============\n" + pformat(processors))

    return processors
