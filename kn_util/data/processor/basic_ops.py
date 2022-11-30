from kn_util.general import registry
import os.path as osp
import h5py

class Delete:

    def __init__(self, from_keys=None) -> None:
        assert from_keys
        self.from_keys = from_keys

    def __call__(self, result):

        for k in self.from_keys:
            del result[k]
        return result


class Rename:

    def __init__(self, from_keys=None, to_keys=None) -> None:
        assert from_keys and to_keys
        self.from_keys = from_keys
        self.to_keys = to_keys

    def __call__(self, result):
        _result = dict()
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            _result[to_key] = result[from_key]
            del result[from_key]
        result.update(_result)

        return result


class Lambda:

    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, result):
        result = self._lambda(result)
        return result


class BatchLambda:
    is_batch_processor = True

    def __init__(self, _lambda) -> None:
        self._lambda = _lambda

    def __call__(self, batch):
        batch = self._lambda(batch)
        return batch


class Collect:

    def __init__(self, from_keys=[]):
        self.from_keys = from_keys

    def __call__(self, result):
        _result = dict()
        for k in self.from_keys:
            _result[k] = result[k]
        return _result

class FileChecker:
    """delete sample if corresponding file doesn't exist"""
    is_batch_processor = True
    def __init__(self, file_template="{video_id}.hdf5") -> None:
        self.file_template = file_template

    def __call__(self, batch):
        ret_batch = []
        for result in batch:
            file_path = self.file_template.format(**result)
            if osp.exists(file_path):
                ret_batch += [file_path]
        return ret_batch

class HDF5Checker:
    """delete sample if key in hdf5 doesn't exist"""
    is_batch_processor = True
    def __init__(self, hdf5_file, key_template="{video_id}") -> None:
        self.hdf5_file = hdf5_file
        self.key_template = key_template
    
    def __call__(self, batch):
        ret_batch = []
        for result in batch:
            with h5py.File(self.hdf5_file, "r") as f:
                k = self.key_template.format(**result)
                if k in f.keys():
                    ret_batch += [result]
        return ret_batch
