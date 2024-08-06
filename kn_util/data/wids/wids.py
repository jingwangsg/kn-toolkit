import base64
import gzip
import hashlib
import io
import os, os.path as osp
import random
import re
import sqlite3
import sys
import uuid
import warnings
from functools import partial
from typing import Any, BinaryIO, Dict, Optional, TypeVar, Union, List
from urllib.parse import quote, urlparse
from loguru import logger
from hashlib import sha256
import dill
from glob import glob

import numpy as np
import torch.distributed as dist

from .wids_dl import download_and_open
from .wids_lru import LRUCache
from .wids_mmtar import MMIndexedTar
from .wids_specs import load_dsdesc_and_resolve, urldir
from .wids_tar import TarFileReader, find_index_file
from .wids_utils import get_file_keys, get_file_meta
from ...dist import get_world_size, is_main_process, broadcast_object_list

from kn_util.utils.io import save_pickle, load_pickle, load_json
from kn_util.utils.multiproc import map_async_with_thread
from kn_util.utils.system import get_strhash, is_valid_file

try:
    from torch.utils.data import Dataset, Sampler
except ImportError:

    class Dataset:
        pass

    class Sampler:
        pass


T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)


def compute_file_md5sum(fname: Union[str, BinaryIO], chunksize: int = 1000000) -> str:
    """Compute the md5sum of a file in chunks.

    Parameters
    ----------
    fname : Union[str, BinaryIO]
        Filename or file object
    chunksize : int, optional
        Chunk size in bytes, by default 1000000

    Returns
    -------
    str
        MD5 sum of the file

    Examples
    --------
    >>> compute_file_md5sum("test.txt")
    'd41d8cd98f00b204e9800998ecf8427e'
    """
    md5 = hashlib.md5()
    if isinstance(fname, str):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(chunksize), b""):
                md5.update(chunk)
    else:
        fname.seek(0)
        for chunk in iter(lambda: fname.read(chunksize), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_file_md5sum(fname: Union[str, BinaryIO], chunksize: int = 1000000) -> str:
    """Compute the md5sum of a file in chunks."""
    md5 = hashlib.md5()
    if isinstance(fname, str):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(chunksize), b""):
                md5.update(chunk)
    else:
        fname.seek(0)
        for chunk in iter(lambda: fname.read(chunksize), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_num_samples(fname):
    ds = IndexedTarSamples(fname)
    return len(ds)


def splitname(fname):
    """Returns the basename and extension of a filename"""
    assert "." in fname, "Filename must have an extension"
    basename, extension = re.match(r"^((?:.*/)?.*?)(\..*)$", fname).groups()
    return basename, extension


# NOTE(ligeng): change to ordered mapping to more flexbile dict
# TODO(ligeng):  submit a PR to fix the mapping issue.
def group_by_key(names):
    """Group the file names by key.

    Args:
        names: A list of file names.

    Returns:
        A list of lists of indices, where each sublist contains indices of files
        with the same key.
    """
    groups = []
    kmaps = {}
    for i, fname in enumerate(names):
        # Ignore files that are not in a subdirectory.
        if "." not in fname:
            logger.warning(f"Warning: Ignoring file {fname} (no '.')")
            continue
        if fname == ".":
            logger.warning(f"Warning: Ignoring the '.' file.")
            continue
        key, ext = splitname(fname)
        if key not in kmaps:
            kmaps[key] = []
        kmaps[key].append(i)
    for k, v in kmaps.items():
        groups.append(v)
    return groups


def default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """A default decoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, stream in sample.items():
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if extension in ["gz"]:
            decompressed = gzip.decompress(stream.read())
            stream = io.BytesIO(decompressed)
            if len(extensions) < 2:
                sample[key] = stream
                continue
            extension = extensions[-2]
        if key.startswith("__"):
            continue
        elif extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            if format == "PIL":
                import PIL.Image

                sample[key] = PIL.Image.open(stream)
            elif format == "numpy":
                import numpy as np

                sample[key] = np.asarray(PIL.Image.open(stream))
            else:
                raise ValueError(f"Unknown format: {format}")
        elif extension == "json":
            import json

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(stream)
        elif extension == "mp":
            import msgpack

            value = stream.read()
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.load(stream)
    return sample


open_itfs = {}


class IndexedTarSamples:
    """A class that accesses samples in a tar file. The tar file must follow
    WebDataset conventions. The tar file is indexed when the IndexedTarSamples
    object is created. The samples are accessed by index using the __getitem__
    method. The __getitem__ method returns a dictionary containing the files
    for the sample. The key for each file is the extension of the file name.
    The key "__key__" is reserved for the key of the sample (the basename of
    each file without the extension). For example, if the tar file contains
    the files "sample1.jpg" and "sample1.txt", then the sample with key
    "sample1" will be returned as the dictionary {"jpg": ..., "txt": ...}.
    """

    def __init__(
        self,
        *,
        path=None,
        stream=None,
        md5sum=None,
        expected_size=None,
        use_mmap=True,
        index_file=find_index_file,
    ):
        assert path is not None or stream is not None

        # Create TarFileReader object to read from tar_file
        self.path = path
        stream = self.stream = stream or open(path, "rb")

        # verify the MD5 sum
        if md5sum is not None:
            stream.seek(0)
            got = compute_file_md5sum(stream)
            assert got == md5sum, f"MD5 sum mismatch: expected {md5sum}, got {got}"
            stream.seek(0)

        # use either the mmap or the stream based implementation
        # NOTE(ligeng): https://stackoverflow.com/questions/11072705/twitter-trends-api-unicodedecodeerror-utf8-codec-cant-decode-byte-0x8b-in-po
        # import gzip
        # print("convert to gzip IO stream")
        # stream = gzip.GzipFile(fileobj=stream)

        if use_mmap:
            self.reader = MMIndexedTar(stream)
        else:
            self.reader = TarFileReader(stream, index_file=index_file)

        # Get list of all files in stream
        all_files = self.reader.names()

        # Group files by key into samples
        self.samples = group_by_key(all_files)
        # print("DEBUG:", list(all_files)[:20])
        # print("DEBUG:", self.samples[:20])

        # check that the number of samples is correct
        if expected_size is not None:
            assert len(self) == expected_size, f"Expected {expected_size} samples, got {len(self)}"

        self.uuid = str(uuid.uuid4())

    def close(self):
        self.reader.close()
        if not self.stream.closed:
            self.stream.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get indexes of files for the sample at index idx
        indexes = self.samples[idx]
        sample = {}
        key = None
        for i in indexes:
            # Get filename and data for the file at index i
            fname, data = self.reader.get_file(i)
            # Split filename into key and extension
            k, ext = splitname(fname)
            # Make sure all files in sample have same key
            key = key or k
            assert key == k
            sample[ext] = data
        # Add key to sample
        sample["__key__"] = key
        return sample

    def __str__(self):
        return f"<IndexedTarSamples-{id(self)} {self.path}>"

    def __repr__(self):
        return str(self)


def hash_localname(dldir="/tmp/_wids_cache"):
    os.makedirs(dldir, exist_ok=True)

    connection = sqlite3.connect(os.path.join(dldir, "cache.db"))
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS cache (url TEXT PRIMARY KEY, path TEXT, checksum TEXT)")
    connection.commit()

    def f(shard):
        """Given a URL, return a local name for the shard."""
        if shard.startswith("pipe:"):
            # uuencode the entire URL string
            hex32 = base64.urlsafe_b64encode(hashlib.sha256(shard.encode()).digest())[:32].decode()
            return os.path.join(dldir, "pipe__" + hex32)
        else:
            # we hash the host and directory components into a 16 character string
            dirname = urldir(shard)
            hex16 = base64.urlsafe_b64encode(hashlib.sha256(dirname.encode()).digest())[:16].decode()
            # the cache name is the concatenation of the hex16 string and the file name component of the URL
            cachename = "data__" + hex16 + "__" + os.path.basename(urlparse(shard).path)
            checksum = None
            cursor.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                (shard, cachename, checksum),
            )
            connection.commit()
            return os.path.join(dldir, cachename)

    return f


def get_cache_path(shard, cachedir):
    """Given a URL, return a local name for the shard."""
    path = urlparse(shard).path
    fname = os.path.basename(path)
    return os.path.join(cachedir, fname)


def cache_localname(cachedir):
    os.makedirs(cachedir, exist_ok=True)

    return partial(get_cache_path, cachedir=cachedir)


def default_localname(dldir="/tmp/_wids_cache"):
    os.makedirs(dldir, exist_ok=True)

    def f(shard):
        """Given a URL, return a local name for the shard."""
        cachename = quote(shard, safe="+-")
        return os.path.join(dldir, cachename)

    return f


class LRUShards:
    """A class that manages a cache of shards. The cache is a LRU cache that
    stores the local names of the shards as keys and the downloaded paths as
    values. The shards are downloaded to a directory specified by dldir.
    The local name of a shard is computed by the localname function, which
    takes the shard URL as an argument. If keep is True, the downloaded files
    are not deleted when they are no longer needed.
    """

    def __init__(self, lru_size, keep=False, localname=default_localname()):
        self.localname = localname
        # the cache contains the local name as the key and the downloaded path as the value
        self.lru = LRUCache(lru_size, release_handler=self.release_handler)
        # keep statistics
        self.reset_stats()

    def reset_stats(self):
        self.accesses = 0
        self.misses = 0

    def __len__(self):
        return len(self.lru)

    def release_handler(self, key, value):
        value.close()

    def clear(self):
        self.lru.clear()

    def get_shard(self, url):
        assert isinstance(url, str)
        self.accesses += 1
        if url not in self.lru:
            local = self.localname(url)
            with download_and_open(url, local) as stream:
                itf = IndexedTarSamples(path=local, stream=stream)
            self.lru[url] = itf
            self.misses += 1
            self.last_missed = True
        else:
            self.last_missed = False
        return self.lru[url]


def interpret_transformations(transformations):
    """Interpret the transformations argument.

    This takes care of transformations specified as string shortcuts
    and returns a list of callables.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]

    result = []

    for transformation in transformations:
        if transformation == "PIL":
            transformation = partial(default_decoder, format="PIL")
        elif transformation == "numpy":
            transformation = partial(default_decoder, format="numpy")
        else:
            assert callable(transformation)
        result.append(transformation)

    return result


def hash_dataset_name(input_string):
    """Compute a hash of the input string and return the first 16 characters of the hash."""
    # Compute SHA256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hash_digest = hash_object.digest()

    # Encode the hash in base64
    base64_encoded_hash = base64.urlsafe_b64encode(hash_digest)

    # Return the first 16 characters of the base64-encoded hash
    return base64_encoded_hash[:16].decode("ascii")


class ShardListDataset(Dataset[T]):
    """An indexable dataset based on a list of shards.

    The dataset is either given as a list of shards with optional options and name,
    or as a URL pointing to a JSON descriptor file.

    Datasets can reference other datasets via `source_url`.

    Shard references within a dataset are resolve relative to an explicitly
    given `base` property, or relative to the URL from which the dataset
    descriptor was loaded.
    """

    def __init__(
        self,
        shards=None,
        dataset_name=None,
        # cache args
        index_cache="/tmp/tar_index/",
        cache_size=int(1e12),
        cache_dir=None,
        lru_size=10,
        localname=None,
        # other args
        transformations="PIL",
        keep=False,
        base=None,
        options=None,
        verbose=False,
        filter_keys: Dict[str, set[str]] = None,
        tar_indexing_on_master=True,
    ):
        """Create a ShardListDataset.

        Args:
            shards: a list of tar files
            cache_size: the number of shards to keep in the cache
            lru_size: the number of shards to keep in the LRU cache
            localname: a function that maps URLs to local filenames

        Note that there are two caches: an on-disk directory, and an in-memory LRU cache.
        """
        os.makedirs(index_cache, exist_ok=True)
        if isinstance(shards, List) and isinstance(shards[0], str):
            # only filenames are given, we need to compute the length
            _shards = None
            key_mapping_by_shard = None

            if not tar_indexing_on_master or is_main_process():

                # load/create file index
                keys_by_file = get_file_keys(shards, cache_dir=index_cache)

                print("Indexing complete")

                key_mapping_by_shard, _shards = get_file_meta(shards, keys_by_file=keys_by_file, filter_keys=filter_keys)

            if tar_indexing_on_master:
                key_mapping_by_shard, shards = broadcast_object_list([key_mapping_by_shard, _shards])
            self.key_mapping_by_shard = key_mapping_by_shard

        # if options is None:
        #     options = {}

        super(ShardListDataset, self).__init__()
        # shards is a list of (filename, length) pairs. We'll need to
        # keep track of the lengths and cumulative lengths to know how
        # to map indices to shards and indices within shards.
        # if isinstance(shards, (str, io.IOBase)):
        #     if base is None and isinstance(shards, str):
        #         shards = osp.expanduser(shards)
        #         base = urldir(shards)
        #     self.base = base
        #     self.spec = load_dsdesc_and_resolve(shards, options=options, base=base)
        #     self.shards = self.spec.get("shardlist", [])
        #     self.dataset_name = self.spec.get("name") or hash_dataset_name(str(shards))
        if isinstance(shards, List):
            if base is None and isinstance(shards, str):
                shards = osp.abspath(osp.expanduser(shards[0][0]))
                base = urldir(shards)
            self.base = base
            self.spec = {"shardlist": [{"url": url, "nsamples": nsamples} for url, nsamples in shards]}
            if dataset_name is not None:
                self.spec["name"] = dataset_name
            self.shards = self.spec.get("shardlist", [])
            self.dataset_name = dataset_name or hash_dataset_name(str(shards))
        else:
            raise NotImplementedError(
                "Only support taking path/url to JSON descriptor file, list of (filename, length) pairs, or list of filenames."
            )
            self.base = None
            self.spec = options
            self.shards = shards
            self.dataset_name = dataset_name or hash_dataset_name(str(shards))

        self.lengths = [shard["nsamples"] for shard in self.shards]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]

        if cache_dir is not None:
            # when a cache dir is explicitly given, we download files into
            # that directory without any changes
            self.cache_dir = cache_dir
            self.localname = cache_localname(cache_dir)
        elif localname is not None:
            # when a localname function is given, we use that
            self.cache_dir = None
            self.localname = localname
        else:
            import getpass

            # when no cache dir or localname are given, use the cache from the environment
            self.cache_dir = os.environ.get("WIDS_CACHE", f"~/.cache/_wids_cache")
            self.cache_dir = osp.expanduser(self.cache_dir)
            self.localname = default_localname(self.cache_dir)

        if verbose or int(os.environ.get("WIDS_VERBOSE", 0)):
            nbytes = sum(shard.get("filesize", 0) for shard in self.shards)
            nsamples = sum(shard["nsamples"] for shard in self.shards)

            logger.info(
                "\t".join(
                    [
                        "[WebShardedList]",
                        f"#shards: {len(self.shards)}",
                        f"base: {self.base}",
                        f"name: {self.spec.get('name')}",
                        f"nfiles: {len(self.shards)}",
                        f"nbytes: {nbytes}",
                        f"samples: {nsamples}",
                        f"cache: {self.cache_dir}",
                    ]
                )
            )
        self.transformations = interpret_transformations(transformations)

        if lru_size > 200:
            warnings.warn("LRU size is very large; consider reducing it to avoid running out of file descriptors")
        self.cache = LRUShards(lru_size, localname=self.localname, keep=keep)

    @property
    def keys(self):
        return self.key2idx.keys()

    def add_transform(self, transform):
        """Add a transformation to the dataset."""
        self.transformations.append(transform)
        return self

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_length

    def get_stats(self):
        """Return the number of cache accesses and misses."""
        return self.cache.accesses, self.cache.misses

    def check_cache_misses(self):
        """Check if the cache miss rate is too high."""
        accesses, misses = self.get_stats()
        if accesses > 100 and misses / accesses > 0.3:
            # output a warning only once
            self.check_cache_misses = lambda: None
            logger.warning("Warning: ShardListDataset has a cache miss rate of {:.1%}%".format(misses * 100.0 / accesses))

    def get_shard(self, index):
        """Get the shard and index within the shard corresponding to the given index."""
        # Find the shard corresponding to the given index.
        shard_idx = np.searchsorted(self.cum_lengths, index, side="right")

        # Figure out which index within the shard corresponds to the
        # given index.
        if shard_idx == 0:
            inner_idx = index
        else:
            inner_idx = index - self.cum_lengths[shard_idx - 1]

        # Get the shard and return the corresponding element.
        desc = self.shards[shard_idx]
        url = desc["url"]

        # ! Deprecated, I don't like this part
        # if url.startswith(("https://", "http://", "gs://", "/", "~")):
        #     # absolute path or url path
        #     url = url
        # else:
        #     # concat relative path
        #     if self.base is None and "base_path" not in self.spec:
        #         raise FileNotFoundError("passing a relative path in shardlist but no base found.")
        #     base_path = self.spec["base_path"] if "base_path" in self.spec else self.base
        #     url = osp.abspath(osp.join(osp.expanduser(base_path), url))

        desc["url"] = url
        try:
            shard = self.cache.get_shard(url)
        except UnicodeDecodeError as e:
            logger.error("UnicodeDecodeError:", desc)
            raise e
        return shard, inner_idx, desc

    def get_by_key(self, key):
        idx = self.key2idx[key]
        return self[idx]

    def __getitem__(self, index):
        """Return the sample corresponding to the given index."""
        shard, inner_idx, desc = self.get_shard(index)

        inner_idx = self.key_mapping_by_shard[desc["url"]][inner_idx]
        sample = shard[inner_idx]

        # Check if we're missing the cache too often.
        self.check_cache_misses()

        sample["__dataset__"] = desc.get("dataset")
        sample["__index__"] = index
        sample["__shard__"] = desc["url"]
        sample["__shardindex__"] = inner_idx

        # Apply transformations
        for transform in self.transformations:
            sample = transform(sample)

        return sample

    def close(self):
        """Close the dataset."""
        self.cache.clear()


def get_funchash(func):
    # serialize and hash a function in python
    if func is None:
        return ""
    func_str = dill.dumps(func)
    return sha256(func_str).hexdigest()[:16]


def _load_keys_by_json(json_file, cache_dir="/tmp/json_index/", keys_filter=None):
    os.makedirs(cache_dir, exist_ok=True)
    filehash = get_strhash(json_file)
    keys_filter_hash = get_funchash(keys_filter)
    cache_file = osp.join(cache_dir, filehash + keys_filter_hash + ".pkl")
    if is_valid_file(cache_file):
        return load_pickle(cache_file)

    try:
        json_dict = load_json(json_file)
    except:
        import debugpy

        debugpy.breakpoint()
    if keys_filter is None:
        json_index = set(json_dict.keys())
    else:
        json_index = keys_filter(json_dict)
    save_pickle(json_index, cache_file)

    return json_index


class ShardListDatasetWithAnnotations(ShardListDataset):

    def __init__(
        self,
        json_files,
        tar_root,
        json_index_cache="/tmp/json_index/",
        keys_filter=None,
        json_indexing_on_master=True,
        *args,
        **kwargs,
    ):
        """ShardListDataset that also loads annotations from a JSON file.
        the annotation will be loaded according to `__key__` of the sample given corresponding __shard__.
        the index order will be the same as the original ShardListDataset.
        """
        keys_in_json = set(osp.basename(json_file).rsplit(".", 1)[0] for json_file in json_files)
        keys_in_tar = set(osp.basename(tar_file).rsplit(".", 1)[0] for tar_file in glob(osp.join(tar_root, "*.tar")))
        keys = keys_in_json & keys_in_tar
        print(f"Found {len(keys)} paris in both json and tar files.")

        json_mapping = {osp.basename(json_file).rsplit(".", 1)[0]: json_file for json_file in json_files}

        # Note: the order of json_files and tar_files should be the same
        json_files = [json_mapping[key] for key in keys]
        tar_files = [osp.join(tar_root, key + ".tar") for key in keys]

        keys_by_json = None
        if not json_indexing_on_master or is_main_process():
            keys_by_json = map_async_with_thread(
                iterable=json_files,
                func=lambda f: _load_keys_by_json(f, cache_dir=json_index_cache, keys_filter=keys_filter),
                verbose=True,
                desc="Gathering keys from json files",
                num_thread=16,
            )

        if json_indexing_on_master:
            keys_by_json = broadcast_object_list([keys_by_json], src=0)[0]

        filter_keys = {tar_file: keys for tar_file, keys in zip(tar_files, keys_by_json)}
        super().__init__(shards=tar_files, filter_keys=filter_keys, *args, **kwargs)
        self.json_files = json_files
        self.json_cache = LRUCache(capacity=self.cache.lru.capacity)

    def get_json(self, shard_key):
        if shard_key not in self.json_cache:
            json_dir = osp.dirname(self.json_files[0])
            json_filename = osp.join(json_dir, shard_key + ".json")
            self.json_cache[shard_key] = load_json(json_filename)
        return self.json_cache[shard_key]

    def get_annotation(self, sample):
        shard_key = osp.basename(sample["__shard__"]).rsplit(".", 1)[0]
        key = sample["__key__"]
        json_data = self.get_json(shard_key)
        return json_data.get(key, None)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        annotation = self.get_annotation(sample)

        return sample, annotation
