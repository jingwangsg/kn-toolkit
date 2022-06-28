import numpy as np


def visual_feature_sampling(visual_feature, max_num_clips, padding=True):
    """
    visual_feature: [num_frame, dim]
    """
    num_clips, dim = visual_feature.shape
    if num_clips <= max_num_clips:
        if padding and max_num_clips > num_clips:
            pad_zeros = np.zeros((max_num_clips - num_clips, dim))
            visual_feature = np.concatenate([visual_feature, pad_zeros], axis=0)
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature
