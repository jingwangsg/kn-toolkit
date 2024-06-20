from huggingface_hub import list_repo_tree
from fnmatch import fnmatch
import os.path as osp
from braceexpand import braceexpand


def _maybe_expand_filenames(filenames):
    if isinstance(filenames, str):
        assert "{" in filenames and "}" in filenames, "Use brace expansion for filenames when providing a string"
        filenames = set(braceexpand(filenames))
    elif isinstance(filenames, list):
        filenames = set(filenames)
    else:
        raise NotImplementedError(f"filenames must be a string or a list, got {type(filenames)}")

    return filenames


def list_files(repo_id, path_in_repo=None, glob_pattern=None, repo_type="dataset", filenames=None):
    assert not (glob_pattern and filenames), "Only one of glob_pattern and filenames can be provided"
    tree = list_repo_tree(repo_id, repo_type=repo_type, path_in_repo=path_in_repo)

    if glob_pattern:
        return sorted([file.path for file in tree if fnmatch(file.path, glob_pattern)])

    if filenames:
        filenames = _maybe_expand_filenames(filenames)
        return sorted([file.path for file in tree if file.path in filenames])

    return sorted([file.path for file in tree])


def list_urls(repo_id, path_in_repo=None, glob_pattern=None, repo_type="dataset", filenames=None):
    filenames = list_files(repo_id, path_in_repo=path_in_repo, glob_pattern=glob_pattern, repo_type=repo_type, filenames=filenames)
    prefix = "https://huggingface.co/datasets/" if repo_type == "dataset" else "https://huggingface.co/"
    base_url = f"{prefix}{repo_id}/resolve/main/"
    return [osp.join(base_url, fn) for fn in filenames]


def validate_files(repo_id, filenames=[]):
    repo_filenames = set(list_files(repo_id, filenames=filenames))
    filenames = set(_maybe_expand_filenames(filenames))

    issubset = filenames.issubset(repo_filenames)
    if not issubset:
        missing_files = filenames - repo_filenames
        print(f"Missing files: {missing_files}")
    return issubset
