import subprocess
import argparse
from kn_util.basic import map_async
from typing import List
import os.path as osp
import os


def parse_args():
    args = argparse.ArgumentParser()
    format = "USER@IP:DIR"
    args.add_argument("from_dir", type=str, help=format)
    args.add_argument("to_dir", type=str, help=format)
    args.add_argument(
        "--async-dir",
        default=False,
        action="store_true",
        help="async download/upload dir",
    )
    args.add_argument("-n", "--chunk-size", type=int, default=100)
    args.add_argument("-P", "--num-process", type=int, default=30)
    args.add_argument(
        "--port",
        type=str,
        default="9999",
    )

    return args.parse_args()


def combine(user=None, ip=None, path=None):
    def _combine(user, ip, path):
        if user and ip:
            return f"{user}@{ip}:{path}"
        else:
            return path

    if isinstance(path, list):
        return _combine(user, ip, " :".join([it_path for it_path in path]))
    else:
        return _combine(user, ip, path)


def parse(s):
    user = None
    ip = None
    dir_path = None
    is_remote = None

    if len(s.split("@")) == 2:
        user, _ = s.split("@")
        ip, dir_path = _.split(":")
        is_remote = True
    else:
        dir_path = s
        is_remote = False

    return user, ip, dir_path, is_remote


def split_path(path):
    return os.path.split(path.rstrip(os.path.sep))


def cmd_list_files(path):
    parent_dir, name = split_path(path)
    return f"cd {parent_dir} && find {name}/ -type f -print0"


def cmd_on_ssh(ip, user, cmd):
    return f"ssh {user}@{ip} '{cmd}'"


def cmd_ssh_relay(from_user, from_ip, to_ip, cmd, port=9999, to_port=22):
    return f"ssh -R 127.0.0.1:{port}:{to_ip}:{to_port} {from_user}@{from_ip} '{cmd}'"


def cmd_rsync(
    to_ip,
    to_user,
    to_path,
    from_ip,
    from_user,
    from_path,
    port=None,
    relative=False,
):
    rsync_args = "-auvP"

    if relative:
        rsync_args += " --relative "

    if port is not None:
        rsync_args += f' -e "ssh -p {port}"'
        to_ip = "127.0.0.1"

    from_path = combine(from_user, from_ip, from_path)
    to_path = combine(to_user, to_ip, to_path)

    return f"rsync {rsync_args} {from_path} {to_path}"


def main(args):
    from_dir = args.from_dir
    to_dir = args.to_dir
    from_user, from_ip, from_path, from_is_remote = parse(from_dir)
    to_user, to_ip, to_path, to_is_remote = parse(to_dir)

    if from_is_remote and to_is_remote:
        print("using remote - remote")
        print("TODO - deprecated now")
        raise NotImplementedError

        def construct_cmd(from_path_holder, to_path_holder):
            return cmd_ssh_relay(
                from_user=from_user,
                from_ip=from_ip,
                to_ip=to_ip,
                port=args.port,
                cmd=cmd_rsync(
                    from_path=from_path_holder,
                    to_ip=to_ip,
                    to_user=to_user,
                    to_path=to_path_holder,
                    port=args.port,
                ),
            )

    else:
        from_platform = "remote" if from_is_remote else "local"
        to_platform = "remote" if to_is_remote else "local"
        print("using {} - {}".format(from_platform, to_platform))

        def construct_cmd(from_path_):
            return cmd_rsync(
                to_ip=to_ip,
                to_user=to_user,
                to_path=to_path,
                from_ip=from_ip,
                from_user=from_user,
                from_path=from_path_,
                relative=args.async_dir,
            )

    if not args.async_dir:
        cmd = construct_cmd(from_path)
        print(cmd)
        subprocess.run(cmd, shell=True)
    else:
        cur_cmd_list_files = cmd_list_files(from_path)
        if from_is_remote:
            cur_cmd_list_files = cmd_on_ssh(from_ip, from_user, cur_cmd_list_files)
        out = subprocess.run(
            cur_cmd_list_files, shell=True, text=True, capture_output=True
        )
        from_files = out.stdout.split("\0")
        from_files = [x for x in from_files if len(x.strip()) > 0]
        from_paths = [osp.join(split_path(from_path)[0], ".", x) for x in from_files]

        assert len(from_files) > 0, "no files found for async dir"
        print("using async dir")

        path_chunks = [
            from_paths[i : i + args.chunk_size]
            for i in range(0, len(from_files), args.chunk_size)
        ]

        def _apply(path_chunk):
            cmd = construct_cmd(path_chunk)
            # print(cmd)
            subprocess.run(cmd, shell=True, capture_output=True)

        map_async(
            func=_apply,
            iterable=path_chunks,
            num_process=args.num_process,
            # test_flag=True,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
