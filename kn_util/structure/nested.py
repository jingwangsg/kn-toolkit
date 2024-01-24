import torch


class NestedTensor(object):

    def __init__(self, tensors, mask, durations):
        self.tensors = tensors
        self.mask = mask
        self.durations = durations

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask, self.durations)

    def decompose(self):
        return self.tensors, self.mask, self.durations

    def subsample(self, stride, start_idx=0):
        # Subsample the video for multi-modal Interaction
        sampled_tensors = [video[start_idx::stride] for video in \
                            torch.split(self.tensors, self.durations, dim=0)]
        sampled_mask = [mask[start_idx::stride] for mask in \
                            torch.split(self.mask, self.durations, dim=0)]

        sampled_durations = [tensor.shape[0] for tensor in sampled_tensors]

        return NestedTensor(torch.cat(sampled_tensors, dim=0), torch.cat(sampled_mask, dim=0), sampled_durations)

    @classmethod
    def from_tensor_list(cls, tensor_list):
        assert tensor_list[0].ndim == 4  # videos
        max_size = tuple(max(s) for s in zip(*[clip.shape for clip in tensor_list]))
        _, c, h, w = max_size

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        # total number of frames in the batch
        durations = [clip.shape[0] for clip in tensor_list]
        nb_images = sum(clip.shape[0] for clip in tensor_list)
        tensor = torch.zeros((nb_images, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((nb_images, h, w), dtype=torch.bool, device=device)
        cur_dur = 0
        for i_clip, clip in enumerate(tensor_list):
            tensor[cur_dur:cur_dur + clip.shape[0], :clip.shape[1], :clip.shape[2], :clip.shape[3],].copy_(clip)
            mask[cur_dur:cur_dur + clip.shape[0], :clip.shape[2], :clip.shape[3]] = False
            cur_dur += clip.shape[0]

        return cls(tensor, mask, durations)

    def __repr__(self):
        return repr(self.tensors)
