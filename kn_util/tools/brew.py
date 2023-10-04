import os
import argparse
import os.path as osp
import subprocess
import glob
from kn_util.basic import map_async


def run_cmd(cmd):
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret


def unique_path(path):
    paths = []
    if path != "":
        paths = list(set([_ for _ in path.split(":") if _ != ""]))

    return ":".join(paths)


def _maybe_patch(executable):
    executable = run_cmd(f"readlink -f {executable}").stdout
    new_fn = f"{executable}_new"
    run_cmd("cp {executable} {new_fn}")
    run_cmd("chmod +rwx {new_fn}")
    runpath = run_cmd(f"readelf -d {new_fn} | grep RUNPATH | grep -oP '\[\K[^]]*'").stdout
    rpath = run_cmd(f"readelf -d {new_fn} | grep RPATH | grep -oP '\[\K[^]]*'").stdout

    ret = run_cmd(f"timeout 1 {executable} --version").stdout
    appended_rpath = "$HOME/homebrew/opt/glibc/lib/"
    rpath = unique_path(f"{runpath}:{rpath}:{appended_rpath}")
    run_cmd(f"patchelf --remove-rpath {new_fn}")
    interpreter_path = "$HOME/opt/glibc/lib64/ld-linux-x86-64.so.2"
    run_cmd(f"patchelf --set-interpreter {interpreter_path} --force-rpath --set-rpath {rpath} {new_fn}")

    bkp_path = f"{executable}.bkp"
    run_cmd(f"mv -f {executable} {bkp_path}")
    run_cmd(f"mv {new_fn} {executable}")


def patch(homebrew_bin):
    all_executable = glob.glob(homebrew_bin + "/*")
    map_async(iterable=all_executable, func=_maybe_patch)


def check_app_available(app):
    ret = run_cmd(f"brew search {app}").stdout
    if ret.startswith(app):
        return True
    else:
        print(f"=> <{app}> not found!")
        print(f"=> Output: {ret}")
        return False

def check_brew_available():
    ret = run_cmd("which brew").stdout
    avail = not (ret == "")

    return avail


def get_dependencies(app):
    ret = run_cmd(f"brew deps -n --missing {app}").stdout  # in typological order
    ret = [_ for _ in ret.split("\n") if _ != ""].reverse()
    return ret


def install(app):
    if not check_app_available(app):
        return

    deps = get_dependencies(app)
    if len(deps) > 0:
        print(f"Installing dependencies: {deps}")
        for dep in deps:
            status = install(dep)

    print(f"=> Installing {app}")
    code = run_cmd(f"brew install {app}").returncode
    if code != 0:
        print(f"=> Failed to install {app} from source! Using --force-bottle instead")
        run_cmd(f"brew install --force-bottle {app}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["patch", "install", "self_install"])
    command = parser.parse_known_args()[0].command
    if command == "patch":
        parser.add_argument("--homebrew_bin", type=str, default=osp.expanduser("~/homebrew/bin"))
        args = parser.parse_args()
        patch(homebrew_bin=args.homebrew_bin)
    elif command == "install":
        parser.add_argument("app", type=str, default="")
        args = parser.parse_args()
        install(args.app)
    elif command == "self_install":
        if check_brew_available():
            print("=> brew already installed!")
            exit(0)
        parser.add_argument("dir", type=str, default="~/")
        args = parser.parse_args()
        run_cmd(f"cd {args.dir} && git clone ")
        if not check_brew_available():
            print("add ~/homebrew/bin and lib/ to ~/.bashrc")
            run_cmd('echo "PATH=/homebrew/bin:$PATH" >> ~/.bashrc')
            run_cmd('echo "LD_LIBRARY_PATH=/homebrew/lib:$LD_LIBRARY_PATH" >> ~/.bashrc')

        

