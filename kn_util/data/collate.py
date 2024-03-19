from torch.utils.data import default_collate


def default_collate_builder(default_keys, list_keys=None):
    def default_collate_wrapped(batch):
        ret = {}
        batch = [{k: v for k, v in item.items() if k in default_keys} for item in batch]
        for key in default_keys:
            ret[key] = default_collate([item[key] for item in batch])
        for key in list_keys or []:
            ret[key] = [item[key] for item in batch]
        return ret

    return default_collate_wrapped
