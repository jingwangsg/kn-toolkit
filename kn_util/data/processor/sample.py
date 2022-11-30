from kn_util.general import registry
import numpy as np

class SequenceSampler:
    def __init__(self, axis=0, max_len=None, stride=None, from_key=None) -> None:
        assert (max_len is not None) ^ (stride is not None)
        assert from_key is not None
        self.max_len = max_len
        self.stride = stride
        self.from_key = from_key
        self.axis = axis

    def __call__(self, result):
        """seq should be a numpy ndarray"""
        seq = result[self.from_key]
        axis = self.axis
        tot_len = seq.shape[axis]
        if self.max_len is not None:
            stride = int(np.ceil((tot_len - 1) / (self.max_len - 1)))
        else:
            stride = self.stride

        dim_shape = len(seq.shape)
        slices = [
            slice(0, seq.shape[_]) if _ != axis else slice(0, tot_len - 1, stride)
            for _ in range(dim_shape)
        ]
        sampled_seq = seq[tuple(slices)]
        slices[axis] = -1
        last = seq[tuple(slices)]
        result[self.from_key + ".sample"] = np.concatenate(
            [sampled_seq, np.expand_dims(last, axis)], axis=axis
        )
        return result
