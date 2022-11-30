from kn_util.general import registry
from kn_util.data import generate_sample_indices
from kn_util.file import load_pickle
import os.path as osp
import h5py
import numpy as np
from kn_util.file import load_hdf5

# import decord as de
import os
from PIL import Image
import glob


class HDF5Loader:

    def __init__(self, hdf5_file, key_template="{}", from_key=None) -> None:
        assert from_key
        self.handle = h5py.File(hdf5_file, "r")
        self.from_key = from_key
        self.key_template = key_template

    def __call__(self, result):
        ind = result[self.from_key]
        item = load_hdf5(self.handle[self.key_template.format(ind)])
        if not isinstance(item, dict):
            item = {self.from_key + ".hdf5": item}
        result.update(item)

        return result


class NumpyLoader:

    def __init__(self, npy_dir=None, path_template="{}.npy", hash_key=None):
        assert npy_dir and hash_key
        self.npy_dir = npy_dir
        self.hash_key = hash_key
        self.path_template = path_template

    def __call__(self, result):
        hash_entry = result[self.hash_key]
        result[self.hash_key + ".npy"] = np.load(
            osp.join(self.npy_dir, self.path_template.format(hash_entry)))
        return result


@registry.register_processor("load_image")
class ImageLoader:

    def __init__(self, from_key=None):
        self.from_key = from_key

    def __call__(self, result):
        path = result[self.from_key]
        if osp.isdir(result[self.from_key]):
            img_paths = glob.glob(osp.join(path, "*.png")) + glob.glob(
                osp.join(path, "*.jpg"))
            imgs = [
                Image.open(img_path).convert("RGB") for img_path in img_paths
            ]
            result[self.from_key + ".img"] = imgs
        else:
            result[self.from_key + ".img"] = Image.open(path).convert("RGB")
        return result


# class DecodeVideoLoader:

#     def __init__(
#         self,
#         max_len=None,
#         stride=None,
#         path_format="{}.avi",
#         from_key="video_id",
#         cache_dir=None,
#     ) -> None:
#         assert (max_len is not None) ^ (stride is not None)
#         self.stride = stride
#         self.max_len = max_len

#         self.path_format = path_format
#         self.from_key = from_key
#         self.cache_dir = cache_dir
#         if cache_dir is not None:
#             os.makedirs(cache_dir, exist_ok=True)

#     def __call__(self, result):
#         video_id = result[self.from_key]
#         if self.cache_dir:
#             cache_file = osp.join(self.cache_dir, f"{video_id}.npy")
#             if osp.exists(cache_file):
#                 arr = np.load(cache_file)
#                 result[self.from_key + "_imgs"] = arr
#                 return result
#         video_path = self.video_format.format(video_id)
#         vr = de.VideoReader(video_path)
#         tot_len = len(vr)

#         sampled_indices = generate_sample_indices(
#             tot_len, self.max_len, self.stride, output_stride=True)
#         arr = vr.get_batch(sampled_indices).asnumpy()

#         if self.cache_dir:
#             np.save(cache_file, arr)

#         result[self.from_key + "_imgs"] = arr
#         return result
