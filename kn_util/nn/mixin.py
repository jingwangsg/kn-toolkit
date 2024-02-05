from loguru import logger
from termcolor import colored
from .mixin_util import _find_module_by_keys, module2tree


class ModelMixin:

    def get_peft_model(
        self,
        keys=[],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
    ):
        if len(keys) == 0:
            logger.info("No keys provided, LoRA disabled.")
            return self
        else:
            logger.info(f"Injecting LoRA into modules: {keys}")

        lora_modules = _find_module_by_keys(self, keys)

        cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=lora_modules,
        )
        return get_peft_model(model=self, peft_config=cfg)

    def set_requires_grad(self, keys=None, requires_grad=True):
        if keys is None:
            for p in self.parameters():
                p.requires_grad = requires_grad
        else:
            for k in keys:
                for n, p in self.named_parameters():
                    if n.startswith(k):
                        p.requires_grad = requires_grad

    def unfreeze(self, keys=None):
        self.set_requires_grad(keys, requires_grad=True)

    def freeze(self, keys=None):
        self.set_requires_grad(keys, requires_grad=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def param_groups(self):
        return {
            "default": {n: p for n, p in self.named_parameters() if p.requires_grad},
        }

    @property
    def num_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def half(self):
        for p in self.parameters():
            p.data = p.data.half()
        return self
    
    def to(self, *args, **kwargs):
        for p in self.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(*args, **kwargs)
        return self

    def pretty_format(self, list_limit=1):
        return module2tree(self, list_limit=list_limit)
