import pandas as pd


class Registry:

    def __init__(self):
        self._map = {}

    def register(self, mapping, name=None):

        def _thunk(cls):
            nonlocal name
            if name is None:
                name = cls.__name__
            if mapping not in self._map:
                self._map[mapping] = {}
            self._map[mapping][name] = cls
            return cls

        return _thunk

    def build(self, mapping, name, *args, **kwargs):
        return self._map[name](*args, **kwargs)

    def build_cls(self, mapping, name, *args, **kwargs):
        return self._map[name]

    def register_object(self, name, obj):
        self.register(name=name, mapping="object")(obj)

    def get_object(self, name):
        self.build_cls(name=name, mapping="object")

    def print_registered(self, mapping):
        # print model zoo as dataframe
        _map = self._map[mapping]
        map_str = {k: v.__module__ + "." + v.__name__ for k, v in _map.items()}
        df = pd.DataFrame(map_str.items(), columns=['name', 'class'])
        df = df.sort_values(by=['name'])
        print(f"Registered {mapping}:")
        print(df.to_markdown(index=False))


registry = Registry()
