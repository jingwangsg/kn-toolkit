# from kn_util.general import registry, get_logger
# import numpy as np
# from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
# import os.path as osp
# import os
# import torch
# import h5py
# import subprocess
# from kn_util.file import save_hdf5, load_hdf5

# log = get_logger(__name__)


# def cache_filter():

#     def out_wrapper(fn):
#         """
#         from_key: [required]
#         cache_args: [optional]
#             cache_dir: [required]
#             hash_key: [optional]
#             load_to_memory: [optional] False
#             verbose: [optional] False
#             overwrite: [optional] False

#         """

#         def wrapper(self, result):
#             cache_args = self.cache_args
#             if not cache_args:  # no cache
#                 return fn(self, result)

#             # use cache
#             if cache_args:
#                 os.makedirs(cache_args["cache_dir"], exist_ok=True)
#             cache_dir = cache_args["cache_dir"]
#             hash_id = result.get(
#                 cache_args.get("hash_key"), hash(result[self.from_key]))
#             cache_file = osp.join(cache_dir, f"{hash_id}.hdf5")
#             verbose = self.cache_args.get("verbose", False)
#             load_to_memory = self.cache_args.get("load_to_memory", False)
#             overwrite = self.cache_args.get("overwrite", False)
#             # no matter whether loaded to memory, inference must run with cache configured

#             if not overwrite and osp.exists(cache_file):  # succesful loaded
#                 try:
#                     with h5py.File(cache_file, "r") as hdf5_handler:
#                         load_item = dict()
#                         for k in hdf5_handler.keys():
#                             load_item.update({k: np.array(hdf5_handler[k])})
#                     if verbose:
#                         log.info(f"cache loaded from {cache_file}")
#                     if load_to_memory:
#                         result.update(load_item)
#                         if verbose:
#                             log.info(f"cache merged to result")
#                     return result
#                 except:
#                     # load failed, run inference
#                     if verbose:
#                         log.info(
#                             f"{cache_file} is invalid and removed, inference will run"
#                         )
#                     subprocess.run(
#                         f"rm -rf {cache_file}",
#                         shell=True)  # delete invalid file

#             # run and save
#             result, load_item = fn(self, result, True)
#             with h5py.File(cache_file, "w") as hdf5_handler:
#                 for k in load_item.keys():
#                     hdf5_handler.create_dataset(k, data=load_item[k])
#             if verbose:
#                 log.info(f"cache stored to {cache_file}")
#             if load_to_memory:
#                 result.update(load_item)
#                 if verbose:
#                     log.info(f"cache merged to result")
#             return result

#         return wrapper

#     return out_wrapper


# class HuggingfaceExtractor:

#     def __init__(self, extractor=None, from_key=None) -> None:
#         assert extractor is not None
#         self.extractor = extractor
#         self.from_key = from_key

#     def __call__(self, result):
#         data = result[self.from_key]
#         result[self.from_key + "_ext"] = self.extractor(data)
#         return result


# class HuggingfaceInference:
#     """inference by sample"""

#     def __init__(self,
#                  model=None,
#                  extractor=None,
#                  from_key=None,
#                  hash_key=None,
#                  cache_hdf5=None,
#                  output_keys=["last_hidden_state"],
#                  load_to_result=True):
#         # hash key as identifier of result[from_key]
#         assert from_key and model and hash_key
#         # to_embeddings=False means for latter use but not load it into memory now
#         log.warn(f"[processor] {self.__class__} may use cuda")
#         self.model = model
#         from transformers.models.clip import CLIPTokenizer
#         self.extractor = extractor

#         self.from_key = from_key
#         self.cache_hdf5 = cache_hdf5
#         self.output_keys = output_keys
#         self.hash_key = hash_key
#         self.load_to_result = load_to_result

#         if cache_hdf5:
#             os.makedirs(osp.dirname(self.cache_hdf5), exist_ok=True)
#             self._build_fr_handler()

#     def _build_fr_handler(self):
#         """build a file reading handler if file exists"""
#         if osp.exists(self.cache_hdf5):
#             self.hdf5_fr = h5py.File(self.cache_hdf5, "r+")
#         else:
#             self.hdf5_fr = None

#     def _close_fr_handler(self):
#         """close file reading handler"""
#         if self.hdf5_fr:
#             self.hdf5_fr.close()
#             self.hdf5_fr = None

#     @torch.no_grad()
#     def __call__(self, result):
#         # split out load_item for convenience for cache_filter
#         # cache_filter is able to decide whether merge load_item into result or latter
#         data = result[self.from_key]
#         identifier = result[self.hash_key]
#         fr = self.hdf5_fr
#         if osp.exists(self.cache_hdf5):
#             try:
#                 if identifier in fr:
#                     if self.load_to_result:
#                         ret_dict = load_hdf5(fr[identifier])
#                         result.update(ret_dict)
#                     return result
#             except:
#                 pass

#         model = self.model.cuda()
#         if self.extractor:
#             inputs = self.extractor(
#                 data,
#                 return_tensors="pt",
#                 max_length=model.config.max_position_embeddings)
#             inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
#         else:
#             inputs = {
#                 "raw_data": data
#             }  # model is not actually a nn.Module here, might be some wrapper
#         outputs = model(**inputs)
#         load_dict = dict()
#         for k in self.output_keys:
#             load_dict[self.from_key + "." +
#                       k] = outputs[k].squeeze(0).cpu().detach().numpy()

#         if self.load_to_result:
#             result.update(load_dict)

#         self._close_fr_handler()
#         if self.cache_hdf5:
#             hdf5_dict = {identifier: load_dict}
#             save_hdf5(
#                 hdf5_dict,
#                 self.cache_hdf5,
#                 compression="gzip",
#                 compression_opts=9)
#         self._build_fr_handler()

#         return result
