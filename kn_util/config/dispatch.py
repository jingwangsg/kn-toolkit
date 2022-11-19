def dispatch_arguments_to_cfgs(cfgs, kwargs):
    for cfg in cfgs:
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                cfg[k] = v