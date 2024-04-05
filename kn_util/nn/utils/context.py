class InitDisabled:
    def __init__(self):
        self.bkp = dict()

    def disable_reset_params(self, cls):
        self.bkp[cls] = getattr(cls, "reset_parameters")
        setattr(cls, "reset_parameters", lambda self: None)

    def enable_reset_params(self):
        for cls, reset_parameters in self.bkp.items():
            setattr(cls, "reset_parameters", reset_parameters)

    def __enter__(self):
        import torch

        self.disable_reset_params(torch.nn.Linear)
        self.disable_reset_params(torch.nn.LayerNorm)
        return self
    
    def __exit__(self, *args, **kwargs):
        import torch
        self.enable_reset_params()
