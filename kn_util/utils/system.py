import subprocess
import socket


def run_cmd(cmd, verbose=False, async_cmd=False):
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
            popen = subprocess.Popen(cmd, shell=True)
            print("=> Running in background")
            return popen


def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]
