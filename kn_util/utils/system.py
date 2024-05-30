import subprocess
import socket
import os.path as osp
import os
from tqdm import tqdm
from contextlib import contextmanager

from .multiproc import map_async_with_thread


@contextmanager
def buffer_keep_open(buffer):
    # ! hack workaround to prevent buffer from closing
    # refer to https://github.com/yt-dlp/yt-dlp/issues/3298
    old_buffer_close = buffer.close
    buffer.close = lambda *_: ...

    yield

    old_buffer_close()


def run_cmd(cmd, verbose=False, async_cmd=False, conda_env=None):
    if conda_env is not None:
        cmd = f"conda run -n {conda_env} {cmd}"

    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in popen.stdout:
            print(line.rstrip().decode("utf-8"))
        popen.wait()
        return popen.returncode
    else:
        if not async_cmd:
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return ret
        else:
            popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return popen


def clear_process(path):
    path = osp.abspath(path)
    processes = run_cmd(f"lsof -n {path} 2>/dev/null | tail -n +2 | awk '{{print $2}}' ").stdout
    cur_pid = os.getpid()
    processes = [int(_.strip()) for _ in processes.split("\n") if _.strip() != ""]
    processes = list(set([pid for pid in processes if pid != cur_pid]))

    run_cmd(f"kill -9 {' '.join(map(str, processes))}")

def clear_process_dir(directory):
    all_files = run_cmd(
        f"find {directory} -type f",
    ).stdout.splitlines()
    all_files = [_.strip() for _ in all_files if _.strip() != ""]
    return map_async_with_thread(iterable=all_files, func=clear_process)


def force_delete(path):
    path = osp.abspath(path)
    clear_process(path)
    run_cmd(f"rm -rf {path}")
    return not osp.exists(path)


def force_delete_dir(directory, quiet=True):
    all_files = run_cmd(
        f"find {directory} -type f",
    ).stdout.splitlines()
    all_files = [_.strip() for _ in all_files if _.strip() != ""]
    return map_async_with_thread(iterable=all_files, func=force_delete, verbose=not quiet)


def get_available_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]
