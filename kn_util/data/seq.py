import numpy as np


def generate_sample_indices(tot_len, max_len=None, stride=None):
    assert (max_len is not None) ^ (stride is not None)
    if max_len is not None:
        stride = int(np.ceil((tot_len - 1) / (max_len - 1)))

    indices = list(range(0, tot_len - 1, stride)) + [tot_len - 1]
    return indices

def slice_by_axis(data, _slice, axis):
    num_axes = len(data.shape)
    slices = tuple([_slice if _ == axis else slice(0, data.shape[_]) for _ in range(num_axes)])
    return data[slices]

def general_sample_sequence(data, axis=0, stride=None, max_len=None, mode="sample"):
    """
    suppose data.shape[axis] = tot_len and we want to devide them into N segments
    a) mode = maxpool / avgpool
       for each segment, we pooling all features into a single feature (using max or avg)
    b) mode = sample
       for each segment, we keep both the head and tail frames
    c) mode = center / random
       for each segment, we select one random frame or its center frame
    """

    tot_len = data.shape[axis]

    if mode in ("maxpool", "avgpool", "center", "random"):
        if stride is None:
            stride = int(np.ceil(tot_len / max_len))
        sampled_frames = []
        for idx in range(0, tot_len, stride):
            cur_frames = slice_by_axis(data, slice(idx, idx + stride), axis=axis)
            if mode == "maxpool":
                sampled_frame = np.max(cur_frames, axis=axis, keepdims=True)
            elif mode == "avgpool":
                sampled_frame = np.mean(cur_frames, axis=axis, keepdims=True)
            elif mode == "center":
                center_idx = stride // 2
                sampled_frame = slice_by_axis(cur_frames, slice(center_idx, center_idx+1), axis=axis)
            elif mode == "random":
                random_idx = np.random.choice(np.arange(stride))
                sampled_frame = slice_by_axis(cur_frames, slice(random_idx, random_idx+1), axis=axis)
            sampled_frames.append(sampled_frame)
        sampled_frames = np.concatenate(sampled_frames, axis=axis)

        return sampled_frames
    
    if mode == "sample":
        if stride is None:
            stride = int(np.ceil((tot_len - 1) / (max_len - 1)))
        sampled_frames = []
        sampled_frames = slice_by_axis(data, slice(0, tot_len - 1, stride), axis=axis)
        tail_frame = slice_by_axis(data, slice(tot_len,None,None), axis=axis)
        sampled_frames = np.concatenate([sampled_frames, tail_frame], axis=axis)

        return sampled_frames