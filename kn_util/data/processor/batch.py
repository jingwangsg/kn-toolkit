from kn_util.general import registry, get_logger
from kn_util.data import general_pad


class SequencePad:
    is_batch_processor = True

    def __init__(self, from_key, return_mask=True, **kwargs) -> None:
        assert from_key
        self.from_key = from_key
        self.kwargs = kwargs
        self.return_mask = return_mask

    def __call__(self, batch) -> None:
        seqs = [e[self.from_key] for e in batch]
        ret = general_pad(seqs, return_mask=self.return_mask, **self.kwargs)
        if self.return_mask:
            padded_val, padded_mask = ret
        else:
            padded_val = ret
        for idx, e in enumerate(batch):
            e[self.from_key + ".pad"] = padded_val[idx]
            if self.return_mask:
                e[self.from_key + ".mask"] = padded_mask[idx]

        return batch
