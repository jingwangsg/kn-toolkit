from ..utils.ssh import SSHTool
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)

    command = parser.parse_known_args()[0].command
    if command == "rtunnel":
        parser.add_argument("remote_node", type=str)
        parser.add_argument("port_dict", nargs="+", type=str)
        args = parser.parse_args()
        port_dict = {}
        for pair in args.port_dict:
            k, v = pair.split(":")
            port_dict[k] = v
        SSHTool.tunnel_remote(port_dict=port_dict, remote_node=args.remote_node)

    elif command == "ltunnel":
        parser.add_argument("remote_node", type=str)
        parser.add_argument("port_dict", nargs="+", type=str)
        args = parser.parse_args()
        port_dict = {}
        for pair in args.port_dict:
            k, v = pair.split(":")
            port_dict[k] = v

        SSHTool.tunnel_local(port_dict=port_dict, remote_node=args.remote_node)

    elif command == "ls":
        SSHTool.list()

    elif command == "jupyter":
        parser.add_argument("remote_node", type=str)
        args = parser.parse_args()
        SSHTool.jupyter(remote_node=args.remote_node)

    else:
        raise NotImplementedError
