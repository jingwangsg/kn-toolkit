import os
import argparse
import os.path as osp
import subprocess
import glob
import sys
from kn_util.utils.multiproc import map_async_with_thread, map_async
from functools import partial
from ..utils.system import run_cmd


def unique_path(path):
    paths = []
    if path != "":
        paths = list(set([_ for _ in path.split(":") if _ != ""]))

    paths = sorted(paths)

    return ":".join(paths)


def _get_readelf(executable, domain="RPATH"):
    return run_cmd(f"readelf -d {executable} | grep {domain} | grep -oP '\[\K[^]]*'").stdout.strip()


def get_appended_rpath(library_names, link_library_homebrew_paths):
    homebrew_paths = list()
    for library_name in library_names:
        if library_name in link_library_homebrew_paths:
            homebrew_paths.append(link_library_homebrew_paths[library_name])
    homebrew_paths = list(set(homebrew_paths))
    if len(homebrew_paths) == 0:
        return None

    appended_rpath = unique_path(":".join(homebrew_paths))

    return appended_rpath


def check_patchable(executable, library_names, link_library_homebrew_paths):
    # prepare appended rpath
    appended_rpath = get_appended_rpath(library_names, link_library_homebrew_paths)

    # readlink and try to append rpath and check if it need to be patched
    if not os.path.exists(executable):
        return False

    if appended_rpath is None:
        return False

    orig_rpath = _get_readelf(executable, domain="RPATH")

    if unique_path(f"{orig_rpath}:{appended_rpath}") == orig_rpath:
        return False

    return True


def get_homebrew_root():
    if osp.exists("/home/linuxbrew/.linuxbrew"):
        return "/home/linuxbrew/.linuxbrew"
    else:
        return osp.expanduser("~/homebrew")


def patch_single(executable, library_names, link_library_homebrew_paths):
    # prepare appended rpath
    appended_rpath = get_appended_rpath(library_names, link_library_homebrew_paths)
    homebrew_root = get_homebrew_root()

    # start patching
    print(f"=> Appended rpath: {appended_rpath}")
    assert os.path.exists(executable), f"{executable} not found!"

    # copy file
    new_fn = f"{executable}_new"
    run_cmd(f"cp {executable} {new_fn}")
    run_cmd(f"chmod +rwx {new_fn}")

    # construct new rpath
    runpath = _get_readelf(new_fn, domain="RUNPATH")
    rpath = _get_readelf(new_fn, domain="RPATH")
    rpath = unique_path(f"{runpath}:{rpath}:{appended_rpath}")

    # patching
    run_cmd(f"LD_LIBRARY_PATH={rpath} patchelf --remove-rpath {new_fn}")
    interpreter_path = f"{homebrew_root}/opt/glibc/lib/ld-linux-x86-64.so.2"
    run_cmd(f"LD_LIBRARY_PATH={rpath} patchelf --set-interpreter {interpreter_path} --force-rpath --set-rpath {rpath} {new_fn}",
            verbose=True)

    # check readelf
    run_cmd(f"readelf -d {new_fn}", verbose=True)

    # backup and mv
    bkp_path = f"{executable}.bkp"
    run_cmd(f"mv -f {executable} {bkp_path}")
    run_cmd(f"mv {new_fn} {executable}")

    print(f"=> Patched {executable}")


def patch(app=None, path=None, need_check=True):
    assert run_cmd("which patchelf").returncode == 0, "patchelf not found!"
    homebrew_root = get_homebrew_root()

    if app is not None:

        def _which(app):
            ret = run_cmd(f"which {app}")
            if ret.returncode != 0:
                print(f"=> <{app}> not found!")
                return None
            else:
                app_path = ret.stdout.strip()
            return app_path

        all_executable = [_which(app) for app in app.split() if _which(app) is not None]
    elif path is not None:
        if "*" in path:
            all_executable = glob.glob(path)
        else:
            all_executable = [path]
    else:
        all_executable = glob.glob(f"{homebrew_root}/bin/*")

    all_executable = list(set([run_cmd(f"readlink -f {executable}").stdout.strip() for executable in all_executable]))

    library_by_app = map_async(
        iterable=all_executable,
        func=get_link_library_single_app,
        desc="Getting link librarys for each Apps",
    )
    library_by_app_mapping = dict(zip(all_executable, library_by_app))
    link_library_homebrew_paths = prepare_link_library_mapping(library_by_app=library_by_app, homebrew_root=homebrew_root)
    print(f"Link librarys found: {list(link_library_homebrew_paths.keys())}")

    # filter
    all_executable = [_ for _ in all_executable if check_patchable(_, library_by_app_mapping[_], link_library_homebrew_paths)]
    print("=> Patchable Apps: ", all_executable)

    if need_check:
        flag = input("=> Patch all? [y/n]")
        if flag.strip() != "y":
            exit(0)

    def _unwrap_patch(executable):

        library_names = library_by_app_mapping[executable]
        patch_single(executable, library_names=library_names, link_library_homebrew_paths=link_library_homebrew_paths)

    map_async(iterable=all_executable, func=_unwrap_patch, desc="Patching Apps")


def check_app_available(app):
    ret = run_cmd(f"brew search --formula {app} | grep {app}").stdout
    similar_apps = [_.strip() for _ in ret.split("\n")]

    if app in similar_apps:
        return True
    else:
        print(f"=> <{app}> not found!")
        print(f"=> Output: {ret}")
        return False


def check_brew_available():
    code = run_cmd("which brew").returncode

    return code == 0


def get_dependencies(app):
    ret = run_cmd(f"brew deps -n --missing {app}").stdout  # in typological order
    ret = [_ for _ in ret.split("\n") if _ != ""]
    return ret


def get_link_library_homebrew(library, homebrew_root="~/homebrew"):
    ret = run_cmd(f"find {homebrew_root} -name \"{library}\"").stdout
    first_line = ret.split("\n")[0]
    homebrew_lib = first_line.rsplit("/", 1)[0]

    return homebrew_lib


def get_link_library_single_app(executable, homebrew_root="~/homebrew"):
    # given a executable, find all link library names
    link_librarys = run_cmd(f"ldd {executable}").stdout
    link_librarys = [_ for _ in link_librarys.split("\n") if "=>" in _]

    link_librarys = [line.split("=>")[0].strip() for line in link_librarys]

    return link_librarys


def prepare_link_library_mapping(library_by_app, homebrew_root="~/homebrew"):
    """prepare mapping from library name to homebrew path"""
    link_library_homebrew = dict()

    # unique lib names
    all_library = []
    for libs in library_by_app:
        all_library += libs
    all_library = list(set(all_library))

    library_homebrew = map_async(
        iterable=all_library,
        func=partial(get_link_library_homebrew, homebrew_root=homebrew_root),
        desc="Getting library homebrew paths for each Apps",
        num_process=30,
    )

    for lib_hb, lib_name in zip(library_homebrew, all_library):
        if lib_hb == "":
            continue
        link_library_homebrew[lib_name] = lib_hb

    return link_library_homebrew


def install(app, post_patch=False):
    if not check_app_available(app):
        return

    deps = get_dependencies(app)
    if len(deps) > 0:
        print(f"Installing dependencies: {deps}")
        for dep in deps:
            status = install(dep)

    if post_patch:
        patch(app=" ".join(deps), need_check=False)

    print(f"=> Installing {app}")
    code = run_cmd(f"brew install {app}", verbose=True)
    if code != 0:
        print(f"=> Failed to install {app} from source! Using --force-bottle instead")
        run_cmd(f"brew install --force-bottle {app}")


def self_install(root_dir):
    if os.geteuid() != 0:
        print("=> not root user! running alternative installation")
        run_cmd(f"cd {root_dir} && git clone --progress https://github.com/Homebrew/brew {root_dir}/homebrew", verbose=True)
    else:
        run_cmd('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', verbose=True)

    run_cmd("brew update --force --quiet", verbose=True)

    if not check_brew_available():
        reply = input("add ~/homebrew/bin and lib/ to ~/.bashrc? [y/n]")
        if reply == "y":
            run_cmd('echo "PATH=/homebrew/bin:$PATH" >> ~/.bashrc')
            run_cmd('echo "LD_LIBRARY_PATH=/homebrew/lib:$LD_LIBRARY_PATH" >> ~/.bashrc')
        else:
            print("=> Please add ~/homebrew/bin and lib/ to ~/.bashrc manually!")

        run_cmd('echo "PATH=/homebrew/bin:$PATH" >> ~/.bashrc')
        run_cmd('echo "LD_LIBRARY_PATH=/homebrew/lib:$LD_LIBRARY_PATH" >> ~/.bashrc')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["patch", "install", "self_install"])
    command = parser.parse_known_args()[0].command
    if command == "patch":
        parser.add_argument("--app", type=str, default=None)
        parser.add_argument("--path", type=str, default=None)
        args = parser.parse_args()
        patch(app=args.app, path=args.path)
    elif command == "install":
        parser.add_argument("apps", nargs="+", type=str)
        parser.add_argument("--post_patch", action="store_true", default=False, help="patch after install")
        args = parser.parse_args()
        for app in args.apps:
            install(app, post_patch=args.post_patch)
    elif command == "self_install":
        if check_brew_available():
            print("=> brew already installed!")
            exit(0)
        parser.add_argument("--root_dir", type=str, default="~/")
        args = parser.parse_args()
        self_install(root_dir=args.root_dir)
