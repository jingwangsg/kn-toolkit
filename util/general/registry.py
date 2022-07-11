class Registry:
    def __init__(self):
        self._cls_dict = dict()

    def register(self, name):
        if isinstance(name, str):
            name = [name]

        def decorator(cls):
            for cur_name in name:
                self._cls_dict[cur_name] = cls
            return cls

        return decorator

    def build(self, name, **kwargs):
        return self._cls_dict[name](**kwargs)

    def build_cls(self, name):
        return self._cls_dict[name]
