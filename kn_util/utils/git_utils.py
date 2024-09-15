import os.path as osp

from .multiproc import map_async_with_thread
from .system import run_cmd


# https://www.zhihu.com/question/269707221/answer/2677167861
def commit(content):
    import git

    repo = git.Repo(search_parent_directories=True)
    try:
        g = repo.git
        g.add("--all")
        res = g.commit("-m " + content)
        print(res)
    except Exception:
        print("no need to commit")


def get_origin_url():
    import git

    repo = git.Repo(search_parent_directories=True)
    remote = repo.remote()
    remote_url = remote.url
    return remote_url


def upload_lfs(files, repo):
    # here repo should be a huggingface repo by default
    url = f"https://huggingface.co/datasets/k-nick/{repo}"
    TMPDIR = osp.expanduser("~/.cache/")
    repo_path = osp.join(TMPDIR, repo)

    run_cmd(f"GIT_LFS_SKIP_SMUDGE=1 git clone {url} {repo_path}")
    map_async_with_thread(iterable=files, func=lambda x: run_cmd(f"mv {x} {repo_path}"))
    
    suffix = set([osp.splitext(x)[1] for x in files])
    run_cmd(f"cd {repo_path} && git lfs track {' '.join(suffix)}")
    run_cmd(f"cd {repo_path} && git add --all")
    run_cmd(f"cd {repo_path} && git commit -m 'upload files'")
    run_cmd(f"cd {repo_path} && git push", verbose=True, async_cmd=True)

    run_cmd(f"rm -rf {osp.join(TMPDIR, repo)}")
