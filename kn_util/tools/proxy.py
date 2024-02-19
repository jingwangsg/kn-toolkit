import argparse
from ..utils.system import run_cmd
import httpx


def build_remote_tunnel(remote, local_port, remote_port):
    run_cmd(
        f"ssh -f -N -R {remote_port}:localhost:{local_port} {remote}",
        async_cmd=True,
        verbose=False,
    )


def add_proxy_args(parser):
    parser.add_argument("--remote", type=str, help="Remote server for tunneling")
    parser.add_argument(
        "--local-port", type=int, help="Local port for tunneling", default=10024
    )
    parser.add_argument(
        "--remote-port", type=int, help="Remote port for tunneling", default=8091
    )


def main():
    parser = argparse.ArgumentParser(description="Proxy")
    add_proxy_args(parser)
    args = parser.parse_args()

    build_remote_tunnel(
        remote=args.remote,
        local_port=args.local_port,
        remote_port=args.remote_port,
    )
