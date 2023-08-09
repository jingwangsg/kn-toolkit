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
    if user and ip:
        return f"{user}@{ip}:{path}"
    else:
        return path


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


def cmd_list_files(path):
    parent_dir, name = os.path.split(path.rstrip(os.path.sep))
    return f"cd {parent_dir} && find {name}/ -type f -print0"


def cmd_on_ssh(ip, user, cmd):
    return f"ssh {user}@{ip} '{cmd}'"


def cmd_ssh_relay(from_user, from_ip, to_ip, cmd, port=9999, to_port=22):
    return f"ssh -R 127.0.0.1:{port}:{to_ip}:{to_port} {from_user}@{from_ip} '{cmd}'"


def cmd_rsync(from_path, to_ip, to_user, to_path, port=None, relative_path=False):
    rsync_args = "-auvP"
    if relative_path:
        rsync_args += " -R"

    if port:
        rsync_args += f' -e "ssh -p {port}"'
        to_ip = "127.0.0.1"

    if isinstance(from_path, List):
        from_path = " ".join(from_path)

    return f"rsync {rsync_args} {from_path} {combine(to_user, to_ip, to_path)}"

def main(args):
    from_dir = args.from_dir
    to_dir = args.to_dir
    from_user, from_ip, from_path, from_is_remote = parse(from_dir)
    to_user, to_ip, to_path, to_is_remote = parse(to_dir)

    use_async = args.async_dir


    if from_is_remote and to_is_remote:
        print("using remote - remote")

        construct_cmd = lambda from_path_holder: cmd_ssh_relay(
            from_user=from_user,
            from_ip=from_ip,
            to_ip=to_ip,
            port=args.port,
            cmd=cmd_rsync(
                from_path=from_path_holder,
                to_ip=to_ip,
                to_user=to_user,
                to_path=to_path,
                port=args.port,
                relative_path=use_async,
            ),
        )

    else:
        print("using local - remote")

        construct_cmd = lambda from_path_holder: cmd_rsync(
            from_path=from_path_holder,
            to_ip=to_ip,
            to_user=to_user,
            to_path=to_path,
            port=args.port,
            relative_path=use_async,
        )

    if not args.async_dir:
        cmd = construct_cmd(from_path)
        print(cmd)
        subprocess.run(cmd, shell=True)
    else:
        cur_cmd_list_files = cmd_list_files(from_path)
        if from_is_remote:
            cur_cmd_list_files = cmd_on_ssh(from_ip, from_user, cur_cmd_list_files)
        out = subprocess.run(cur_cmd_list_files, shell=True, text=True, capture_output=True)
        from_files = out.stdout.split("\0")
        
        assert len(from_files) > 0, "no files found for async dir"
        print("using async dir")
        from_file_chunks = [
            from_files[i : i + args.chunk_size]
            for i in range(0, len(from_files), args.chunk_size)
        ]

        map_async(
            func=lambda from_path_holder: subprocess.run(
                construct_cmd(from_path_holder), shell=True, capture_output=True
            ),
            iterable=from_file_chunks,
            num_process=args.num_process,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
