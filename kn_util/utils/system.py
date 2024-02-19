import subprocess
import socket
import os.path as osp


def run_cmd(cmd, verbose=False, async_cmd=False):
    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in popen.stdout:
            print(line.rstrip().decode("utf-8"))
        popen.wait()
        return popen.returncode
    else:
        if not async_cmd:
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return ret
        else:
            popen = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return popen


def clear_process(path):
    dirname, filename = osp.dirname(path), osp.basename(path)
    # print(run_cmd(f"lsof +D {dirname} | grep {path}").stdout)
    run_cmd(f"lsof +D {dirname} | grep {filename} | awk '{{print $2}}' | xargs kill -9")


def force_delete(path):
    path = osp.abspath(path)
    clear_process(path)
    run_cmd(f"rm -rf {path}")
    return not osp.exists(path)


def force_delete_dir(directory):
    all_files = run_cmd(
        f"find {directory} -type f",
    ).splitlines()
    all_files = [_.strip() for _ in all_files if _.strip() != ""]
    return (force_delete(file) for file in all_files)


def get_available_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]
