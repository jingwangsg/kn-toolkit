class Registry:
    def __init__(self):
        self._cls_dict = dict()

    def register(self, name):
        def decorator(cls):
            self._cls_dict[name] = cls
            return cls

        return decorator

    def build(self, name, **kwargs):
        return self._cls_dict[name](**kwargs)
