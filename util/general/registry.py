class Registry:
    def __init__(self):
        self._cls_dict = dict()

    def register(self, name=None):
        def decorator(cls):
            # if name is None:
            #     name = cls.__name__
            self._cls_dict[name] = cls
            return cls

        return decorator

    def build(self, name, **kwargs):
        return self._cls_dict[name](**kwargs)

    def build_cls(self, name):
        return self._cls_dict[name]
