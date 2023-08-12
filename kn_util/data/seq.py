import numpy as np


def generate_sample_indices(tot_len, max_len=None, stride=None):
    """helper function to generate indices for sampling a sequence
    """
    assert (max_len is not None) ^ (stride is not None)
    if max_len is not None:
        stride = int(np.ceil((tot_len - 1) / (max_len - 1)))

    indices = list(range(0, tot_len - 1, stride)) + [tot_len - 1]
    return indices


def slice_by_axis(data, _slice, axis):
    """ slice a tensor by axis
    """
    num_axes = len(data.shape)
    slices = tuple([_slice if _ == axis else slice(0, data.shape[_]) for _ in range(num_axes)])
    return data[slices]


def reduce_segment(data, st_idx, ed_idx, axis=0, mode="avgpool"):
    """ reduce a segment of frames into a single frame feature
    now we support 5 modes:
    a) maxpool: max pooling
    b) avgpool: average pooling
    c) center: select the center frame
    d) random: select a random frame
    e) tail: select the tail frame
    """
    span = ed_idx - st_idx
    if st_idx == ed_idx:
        cur_frames = slice_by_axis(data, slice(st_idx, st_idx + 1), axis=axis)
        return cur_frames
    cur_frames = slice_by_axis(data, slice(st_idx, ed_idx), axis=axis)
    if mode == "maxpool":
        sampled_frame = np.max(cur_frames, axis=axis, keepdims=True)
    elif mode == "avgpool":
        sampled_frame = np.mean(cur_frames, axis=axis, keepdims=True)
    elif mode == "center":
        # for each segment, we select the center frame
        center_idx = span // 2
        sampled_frame = slice_by_axis(cur_frames, slice(center_idx, center_idx + 1), axis=axis)
    elif mode == "random":
        # for each segment, we randomly select one frame
        random_idx = np.random.choice(np.arange(ed_idx - st_idx))
        sampled_frame = slice_by_axis(cur_frames, slice(random_idx, random_idx + 1), axis=axis)
    elif mode == "tail":
        # for each segment, we select the tail frame
        tail_idx = ed_idx - 1
        sampled_frame = slice_by_axis(cur_frames, slice(tail_idx, tail_idx + 1), axis=axis)
    return sampled_frame


def sample_sequence_general(data, axis=0, stride="round", seq_len=None, mode="avgpool"):
    """sample a sequence into fixed length segments
    
    Constant Stride Mode:
    1) when stride is an integer, we will sample the sequence with a constant stride
    2) when stride is "constant" and seq_len is given, we will sample the sequence with a calculated stride

    Rounding Index Mode:
    we will round the index to the nearest integer, and then sample the sequence

    Args:
        data: a array of frames with shape [num_frames, ...]
        axis: the axis to sample on
        stride: the stride to sample on, can be an int or "round" or "constant" when max_len is given
        seq_len: the expected length of sampled sequence
        mode: the mode to *pool segments*, can be "maxpool", "avgpool", "center", "random", "tail"

    Returns:
        ret_frames: a array of frames with shape [num_frames, ...]
    """

    tot_len = data.shape[axis]
    ret_frames = []
    # stride = "constant"
    if stride == "constant" or isinstance(stride, int):
        # length cannot be fixed if stride is a constant
        if stride == "constant":
            assert seq_len is not None
            stride = np.ceil(tot_len / seq_len)
            # in this case, final length cannot be gaurenteed
            #! if a fixed length is required, please use "round" mode
        idxs = np.arange(0, tot_len, stride, dtype=int)

    # stride = "round"
    # length will be fixed to max_len
    else:
        assert seq_len, "seq_len must be given when stride is 'round'"
        idxs = np.arange(0, seq_len + 1, 1.0) / seq_len * tot_len
        idxs = np.round(idxs).astype(np.int32)
        idxs[idxs >= tot_len] = tot_len - 1

    for i in range(len(idxs) - 1):
        st_idx = idxs[i]
        ed_idx = idxs[i + 1]
        sampled_frame = reduce_segment(data, st_idx, ed_idx, axis=axis, mode=mode)
        ret_frames.append(sampled_frame)
    ret_frames = np.concatenate(ret_frames, axis=axis)
    return ret_frames
