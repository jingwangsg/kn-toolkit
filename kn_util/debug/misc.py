from ..basic import registry

class SignalContext:
    # https://book.pythontips.com/en/latest/context_managers.html
    def __init__(self, signal, valid):
        self.signal = signal
        self.valid = valid
    
    def __enter__(self):
        registry.register_object(self.signal, self.valid)
    
    def __exit__(self, type, value, traceback):
        registry.destry_object(self.signal)

def minitensor(tensor, size=5, use_first_batch=True):
    num_dim = len(tensor.shape)
    slices = [slice(0, tensor.shape[i]) if tensor.shape[i] < size else slice(0, size) for i in range(num_dim)]
    if use_first_batch:
        slices[0] = 0
    return tensor[tuple(slices)]