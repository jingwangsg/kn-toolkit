from torch.utils.data import Sampler
import random


class ConcatSampler(Sampler):
    def __init__(self, samplers, shuffle=False, seed=0):
        self.samplers = samplers

        self.sampler_lengths = sampler_lengths = [len(sampler) for sampler in samplers]
        self.total_length = sum(sampler_lengths)

        sampler_schedule = []
        for i, length in enumerate(sampler_lengths):
            sampler_schedule += [i] * length

        self.rng = random.Random(seed)

        if shuffle:
            self.rng.shuffle(sampler_schedule)

        self.sampler_schedule = sampler_schedule

        # convert all samplers to iters
        self.samplers = [iter(sampler) for sampler in samplers]

    def __iter__(self):
        for i in self.sampler_schedule:
            yield next(self.samplers[i])

    def __len__(self):
        return self.total_length

    def set_epoch(self, epoch):
        for sampler in self.samplers:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
