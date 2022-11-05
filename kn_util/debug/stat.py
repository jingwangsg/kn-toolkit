import numpy as np

class Statistics:
    vals = dict()
    @classmethod
    def update(cls, name, val):
        if name not in cls.vals:
            cls.vals[name] = []
        cls.vals[name] += [val]

    @classmethod
    def compute(cls, name, val, op):
        if op == "max":
            return np.max(cls.vals[name])
        elif op == "avg":
            return np.mean(cls.vals[name])
        elif op == "min":
            return np.min(cls.vals[name])