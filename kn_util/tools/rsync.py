import argparse
import subprocess
import os, os.path as osp
from datetime import datetime

from ..utils.system import run_cmd
from ..utils.multiproc import map_async, map_async_with_thread


def run_cmd_remote_maybe(cmd, hostname=None):
    def cmd_on_ssh(hostname, cmd):
        return f"ssh {hostname} '{cmd}'"

    cur_hostname = run_cmd("hostname").stdout.strip()
    if hostname is not None and hostname != cur_hostname:
        cmd = cmd_on_ssh(hostname, cmd)
    return run_cmd(cmd)


def split_path(path):
    return os.path.split(path.rstrip(os.path.sep))


def cmd_list_files(path, min_size=None, max_size=None):
    cmd = f"$HOME/homebrew/bin/fd --type f --base-directory {path} -H -I"
    if min_size is not None:
        cmd += f" --size +{min_size}"
    if max_size is not None:
        cmd += f" --size -{max_size}"
    return cmd


def cmd_get_path(path):
    return f"readlink -f {path}"


def cmd_ssh_relay(from_user, from_ip, to_ip, cmd, port=9999, to_port=22):
    return f"ssh -R 127.0.0.1:{port}:{to_ip}:{to_port} {from_user}@{from_ip} '{cmd}'"


def check_hostname_available(hostname):
    returncode = run_cmd_remote_maybe("echo 1", hostname).returncode
    return returncode == 0


def prepare_path_for_rsync(hostname=None, path=None):
    """path can be a list of paths or a single path"""

    def _combine(hostname, path):
        return f"{hostname}:{path}" if hostname is not None else path

    is_remote = hostname is not None

    path_delimiter = " :" if is_remote else " "

    if isinstance(path, list):
        return _combine(hostname, path_delimiter.join([f"'{it_path}'" for it_path in path]))
    else:
        return _combine(hostname, path)


def get_last_modified(path, hostname):
    if run_cmd_remote_maybe(f"find {path}", hostname).returncode != 0:
        return datetime.fromtimestamp(0)

    # only support linux so far
    last_modified = (
        run_cmd_remote_maybe(f'find {path} -type f -printf "%-.22T+ %M %n %-8u %-8g %8s %Tx %.8TX %p\n" | sort | tail -1', hostname)
        .stdout.strip()
        .split(" ")[0]
    )
    # example format 2024-05-06+18:42:28.58
    last_modified = datetime.strptime(last_modified, "%Y-%m-%d+%H:%M:%S.%f")

    return last_modified


def cmd_rsync(
    to_host,
    to_path,
    from_host,
    from_path,
    relative=False,
    remove_source_files=False,
    exclude=None,
):
    rsync_args = "-auvP"

    if relative:
        rsync_args += " --relative "

    if remove_source_files:
        rsync_args += " --remove-source-files "

    if exclude is not None:
        rsync_args += f" --exclude={exclude} "

    from_path_for_rsync = prepare_path_for_rsync(hostname=from_host, path=from_path)
    to_path_for_rsync = prepare_path_for_rsync(hostname=to_host, path=to_path)

    return f"rsync {rsync_args} {from_path_for_rsync} {to_path_for_rsync}"


def parse(s):
    dir_path = None
    hostname = None

    if len(s.split(":")) == 2:
        hostname, dir_path = s.split(":")
    else:
        dir_path = s

    if hostname is not None:
        dir_path = run_cmd_remote_maybe(cmd_get_path(dir_path), hostname).stdout.strip()
    else:
        dir_path = osp.realpath(dir_path)

    if s.endswith("/") and not dir_path.endswith("/"):
        dir_path += "/"  # prevent readlink -f from removing the last slash

    return hostname, dir_path


class RsyncTool:

    @classmethod
    def delete(cls, dir_path):
        empty_dir = osp.expanduser("~/.empty")
        run_cmd(f"rm -rf {empty_dir} && mkdir {empty_dir}")
        run_cmd(f"rsync --delete-before --force -r {empty_dir} {dir_path}", verbose=True)
        run_cmd(f"rm -rf {empty_dir}")
        run_cmd(f"rm -rf {dir_path}")

    @classmethod
    def launch_rsync(
        cls,
        from_addr,
        to_addr,
        async_dir=False,
        path_filter=lambda x: True,
        chunk_size=100,
        num_thread=30,
        **rsync_kwargs,
    ):
        # async_dir: rsync all files in from_addr to to_addr asychronously
        # path_filter: custom function for filtering files, or all files in dir will be rsynced

        from_host, from_path = parse(from_addr)
        to_host, to_path = parse(to_addr)

        if from_host is not None:
            assert check_hostname_available(from_host), f"hostname {from_host} not available"
        if to_host is not None:
            assert check_hostname_available(to_host), f"hostname {to_host} not available"

        num_remotes = (from_host is not None) + (to_host is not None)
        if num_remotes == 2:
            raise NotImplementedError("remote to remote rsync not supported")

        from_mode = "remote" if from_host is not None else "local"
        to_mode = "remote" if to_host is not None else "local"

        print("=> Using {} - {}".format(from_mode, to_mode))

        if not async_dir:
            cmd = cmd_rsync(
                to_host=to_host,
                to_path=to_path,
                from_host=from_host,
                from_path=from_path,
                **rsync_kwargs,
            )
            print(cmd)
            subprocess.run(cmd, shell=True)
        else:
            # list all files recursively in from_path
            def list_files(from_path, min_size=None, max_size=None):
                cmd = cmd_list_files(from_path, min_size=min_size, max_size=max_size)

                # out = run_cmd(cmd).stdout.strip()
                out = run_cmd_remote_maybe(cmd, from_host).stdout.strip()

                from_files = out.split("\n")
                from_files = [x for x in from_files if len(x.strip()) > 0 and path_filter(x)]
                from_paths = [osp.join(from_path, ".", fn) for fn in from_files]
                return from_paths

            # construct as relative path for --relative rsync

            print("=> using async dir")

            rsync_kwargs["to_host"] = to_host
            rsync_kwargs["to_path"] = to_path
            rsync_kwargs["from_host"] = from_host

            file_small = list_files(from_path, max_size="512m")
            cls.rsync_in_chunk(file_small, chunk_size=chunk_size, num_thread=num_thread, desc="Rsync Small", **rsync_kwargs)

            files_large = list_files(from_path, max_size="10g", min_size="511m")
            cls.rsync_in_chunk(files_large, chunk_size=1, num_thread=num_thread, desc="Rsync Large", **rsync_kwargs)

            files_extreme_large = list_files(from_path, min_size="9g")
            continue_large = input(f"Continue with extreme large files? (>=9g)\n{files_extreme_large}\n(y/n)")
            if continue_large == "y":
                cls.rsync_in_chunk(files_extreme_large, chunk_size=1, num_thread=num_thread, desc="Rsync Extreme Large", **rsync_kwargs)

    @classmethod
    def rsync_in_chunk(cls, paths, chunk_size, num_thread=32, desc="Rsync", **rsync_kwargs):
        # construct chunks
        path_chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]

        def _apply(path_chunk):
            cmd = cmd_rsync(from_path=path_chunk, relative=True, **rsync_kwargs)
            run_cmd(cmd, verbose=False, async_cmd=False)

        map_async_with_thread(
            func=_apply,
            iterable=path_chunks,
            num_thread=num_thread,
            desc=desc,
        )


def main():
    parser = argparse.ArgumentParser()
    format_help = "HOSTNAME:DIR"
    parser.add_argument("command", type=str)

    command = parser.parse_known_args()[0].command
    if command == "launch":
        parser.add_argument("from_dir", type=str, help=format_help)
        parser.add_argument("to_dir", type=str, help=format_help)
        parser.add_argument("--async-dir", default=False, action="store_true", help="async download/upload dir")
        parser.add_argument("--chunk-size", default=30, type=int, help="chunk size for async dir")
        parser.add_argument("-n", "--num-thread", type=int, default=16)
        parser.add_argument("--paths", type=str, help="python expression to generate paths", default=None)

        args = parser.parse_args()

        paths = eval(args.paths) if args.paths is not None else None
        path_filter = (lambda x: True) if paths is None else (lambda x: x in paths)

        RsyncTool.launch_rsync(
            from_addr=args.from_dir,
            to_addr=args.to_dir,
            async_dir=args.async_dir,
            path_filter=path_filter,
            chunk_size=args.chunk_size,
            num_thread=args.num_thread,
        )

    elif command == "delete":
        parser.add_argument("dir_path", type=str, help="dir to delete")
        args = parser.parse_args()

        RsyncTool.delete(dir_path=args.dir_path)
    else:
        raise NotImplementedError
