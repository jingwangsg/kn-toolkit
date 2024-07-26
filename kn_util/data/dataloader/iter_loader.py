import time


class IterLoader:

    def __init__(self, dataloader, num_iters=None):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0
        self.num_iters = num_iters

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch"):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data
    
    def __iter__(self):
        cnt = 0
        while cnt < self.num_iters:
            yield self.__next__()
            cnt += 1

    def __len__(self):
        return self.num_iters
