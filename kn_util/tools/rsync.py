from ..utils.rsync import RsyncTool
import argparse

if __name__ == "__main__":
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
        parser.add_argument("-n", "--chunk-size", type=int, default=100)
        parser.add_argument("-P", "--num-process", type=int, default=30)

        args = parser.parse_args()

        RsyncTool.launch_rsync(from_addr=args.from_dir,
                               to_addr=args.to_dir,
                               async_dir=args.async_dir,
                               chunk_size=args.chunk_size,
                               num_process=args.num_process)
    elif command == "delete":
        parser.add_argument("dir_path", type=str, help="dir to delete")
        args = parser.parse_args()

        RsyncTool.delete(dir_path=args.dir_path)
    else:
        raise NotImplementedError
