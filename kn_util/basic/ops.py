def add_prefix_dict(cur_dict, prefix=""):
    return {prefix+k: v for k,v in cur_dict.items()}