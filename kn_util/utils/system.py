import subprocess

def run_cmd(cmd, verbose=False):
    if verbose:
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in popen.stdout:
            print(line.rstrip().decode("utf-8"))
        popen.wait()
        return popen.returncode

    else:
        ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return ret