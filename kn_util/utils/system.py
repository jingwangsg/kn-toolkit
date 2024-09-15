import io
import os
import os.path as osp
import socket
import subprocess
import traceback
from contextlib import contextmanager
from hashlib import sha256

import psutil

from .mail import send_email
from .multiproc import map_async_with_thread


@contextmanager
def buffer_keep_open(buffer):
    # ! hack workaround to prevent buffer from closing
    # refer to https://github.com/yt-dlp/yt-dlp/issues/3298
    old_buffer_close = buffer.close
    buffer.close = lambda *_: ...

    yield

    old_buffer_close()


def run_cmd(cmd, verbose=False, async_cmd=False, conda_env=None, fault_tolerance=False):
    if conda_env is not None:
        cmd = f"conda run -n {conda_env} {cmd}"

    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in popen.stdout:
            print(line.rstrip().decode("utf-8"))
        popen.wait()
        if popen.returncode != 0 and not fault_tolerance:
            raise RuntimeError(f"Failed to run command: {cmd}\nERROR {popen.stderr}\nSTDOUT{popen.stdout}")
        return popen.returncode
    else:
        if not async_cmd:
            # decode bug fix: https://stackoverflow.com/questions/73545218/utf-8-encoding-exception-with-subprocess-run
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="cp437")
            if ret.returncode != 0 and not fault_tolerance:
                raise RuntimeError(f"Failed to run command: {cmd}\nERROR {ret.stderr}\nSTDOUT{ret.stdout}")
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

    if len(processes) > 0:
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


def get_current_command():
    current_process_pid = os.getpid()
    current_process = psutil.Process(current_process_pid)
    command_line = current_process.cmdline()

    return " ".join(command_line)


def get_hostname():
    return socket.gethostname()


def get_pid():
    return os.getpid()


def get_exception_handler(to_email=False, to_file=None):
    def exception_handler(exc_type, exc_value, exc_traceback):
        nonlocal to_file, to_email
        handler = io.StringIO()
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=handler)
        hostname = get_hostname()
        cmd = get_current_command()
        text = "\n".join(
            [
                cmd,
                "=" * 30,
                handler.getvalue(),
            ]
        )
        if to_email:
            send_email(to_addr="kningtg@gmail.com", subject=f"Error on {hostname}", text=text)
        if to_file is not None:
            nm, ext = to_file.rsplit(".", 1)
            to_file = f"{nm}.pid{get_pid()}.{ext}"
            with open(to_file, "w") as f:
                f.write(text)

    return exception_handler


def get_strhash(s):
    return sha256(s.encode()).hexdigest()[:16]


def get_filehash(file, first_n_bytes=4096 * 10):
    import hashlib

    hash_md5 = hashlib.md5()
    byte_cnt = 0
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            byte_cnt += 4096
            if byte_cnt >= first_n_bytes:
                break
    
    return hash_md5.hexdigest()


def is_valid_file(file):
    return osp.exists(file) and osp.getsize(file) > 0
