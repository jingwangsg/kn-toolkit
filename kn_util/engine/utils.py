def update_optimizer(optimizer, lr_schedule_values=None, wd_schedule_values=None, it=None):
    """Update optimizer learning rate and weight decay."""
    if lr_schedule_values is not None or wd_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[it]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[it]
