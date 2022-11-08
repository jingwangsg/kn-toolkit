from ..general import global_registry

class SignalContext:
    # https://book.pythontips.com/en/latest/context_managers.html
    def __init__(self, signal, value=True):
        self.signal = signal
        self.value = value
    
    def __enter__(self):
        global_registry.register_object(self.signal, self.value)
    
    def __exit__(self, type, value, traceback):
        global_registry.destry_object(self.signal)