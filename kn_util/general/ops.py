def set_attribute(obj, attr_dict):
    for k, v in attr_dict.item():
        setattr(obj, k, v)