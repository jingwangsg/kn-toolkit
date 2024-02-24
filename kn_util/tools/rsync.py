import argparse
import subprocess
import os
from ..utils.multiproc import map_async, map_async_with_thread
import os.path as osp
from ..utils.system import run_cmd


def split_path(path):
    return os.path.split(path.rstrip(os.path.sep))


def cmd_list_files(path, size="+1g"):
    return f"cd {path} && fd --type f --size {size}"


def cmd_get_path(path):
    return f"readlink -f {path}"


def cmd_on_ssh(hostname, cmd):
    return f"ssh {hostname} '{cmd}'"


def cmd_ssh_relay(from_user, from_ip, to_ip, cmd, port=9999, to_port=22):
    return f"ssh -R 127.0.0.1:{port}:{to_ip}:{to_port} {from_user}@{from_ip} '{cmd}'"


def parse(s):
    dir_path = None
    hostname = None

    if len(s.split(":")) == 2:
        hostname, dir_path = s.split(":")
    else:
        dir_path = s

    if hostname is not None:
        dir_path = run_cmd(cmd_on_ssh(hostname, cmd_get_path(dir_path))).stdout.strip()
    else:
        dir_path = osp.realpath(dir_path)

    if s.endswith("/") and not dir_path.endswith("/"):
        dir_path += "/"  # prevent readlink -f from removing the last slash

    return hostname, dir_path


class RsyncTool:

    @staticmethod
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

    @classmethod
    def cmd_rsync(
        cls,
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

        from_path_for_rsync = cls.prepare_path_for_rsync(hostname=from_host, path=from_path)
        to_path_for_rsync = cls.prepare_path_for_rsync(hostname=to_host, path=to_path)

        return f"rsync {rsync_args} {from_path_for_rsync} {to_path_for_rsync}"

    @staticmethod
    def check_hostname_available(hostname):
        returncode = run_cmd(cmd_on_ssh(hostname, "echo 1")).returncode
        return returncode == 0

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
            assert cls.check_hostname_available(from_host), f"hostname {from_host} not available"
        if to_host is not None:
            assert cls.check_hostname_available(to_host), f"hostname {to_host} not available"

        num_remotes = (from_host is not None) + (to_host is not None)
        if num_remotes == 2:
            raise NotImplementedError("remote to remote rsync not supported")

        from_mode = "remote" if from_host is not None else "local"
        to_mode = "remote" if to_host is not None else "local"

        print("=> Using {} - {}".format(from_mode, to_mode))

        def construct_cmd(from_path_):
            return cls.cmd_rsync(
                to_host=to_host,
                to_path=to_path,
                from_host=from_host,
                from_path=from_path_,
                relative=async_dir,
                **rsync_kwargs,
            )

        if not async_dir:
            cmd = construct_cmd(from_path)
            print(cmd)
            subprocess.run(cmd, shell=True)
        else:
            # list all files recursively in from_path
            def list_files(from_path, size="+1g"):
                cmd = cmd_list_files(from_path, size=size)
                if from_mode == "remote":
                    cmd = cmd_on_ssh(from_host, cmd)

                out = run_cmd(cmd).stdout.strip()
                from_files = out.split("\n")
                from_files = [x for x in from_files if len(x.strip()) > 0 and path_filter(x)]
                from_paths = [osp.join(from_path, ".", fn) for fn in from_files]
                return from_paths

            # construct as relative path for --relative rsync

            print("=> using async dir")

            def rsync_in_chunk(paths, chunk_size, desc="Rsync"):
                # construct chunks
                path_chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]

                def _apply(path_chunk):
                    cmd = construct_cmd(path_chunk)
                    run_cmd(cmd, verbose=True)

                map_async_with_thread(
                    func=_apply,
                    iterable=path_chunks,
                    num_thread=num_thread,
                    desc=desc,
                )

            file_small = list_files(from_path, size="-512m")
            rsync_in_chunk(file_small, chunk_size=chunk_size, desc="Rsync Small")

            files_large = list_files(from_path, size="+512m")
            rsync_in_chunk(files_large, chunk_size=1, desc="Rsync Large")


def main():
    parser = argparse.ArgumentParser()
    format_help = "HOSTNAME:DIR"
    parser.add_argument("command", type=str)

    command = parser.parse_known_args()[0].command
    if command == "launch":
        parser.add_argument("from_dir", type=str, help=format_help)
        parser.add_argument("to_dir", type=str, help=format_help)
        parser.add_argument(
            "--async-dir",
            default=False,
            action="store_true",
            help="async download/upload dir",
        )
        parser.add_argument("-n", "--chunk-size", type=int, default=10)
        parser.add_argument("-P", "--num-thread", type=int, default=30)

        args = parser.parse_args()

        RsyncTool.launch_rsync(
            from_addr=args.from_dir,
            to_addr=args.to_dir,
            async_dir=args.async_dir,
            chunk_size=args.chunk_size,
            num_thread=args.num_thread,
        )
    elif command == "delete":
        parser.add_argument("dir_path", type=str, help="dir to delete")
        args = parser.parse_args()

        RsyncTool.delete(dir_path=args.dir_path)
    else:
        raise NotImplementedError
