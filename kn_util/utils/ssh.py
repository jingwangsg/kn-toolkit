from ..utils.system import run_cmd


class SSHTool:

    @staticmethod
    def list():
        run_cmd("netstat -tulpn | grep ssh", verbose=True)

    @staticmethod
    def _get_hostname():
        hostname = run_cmd("hostname").stdout.strip()
        return hostname

    @staticmethod
    def remote_list(remote_node):
        run_cmd(f"ssh {remote_node} 'netstat -tulpn | grep ssh'", verbose=True)

    @classmethod
    def jupyter(cls, remote_node):
        hostname = cls._get_hostname()
        node_idx = int(hostname[4:])
        port = f"89{node_idx:02d}"
        print(f"=> Launching Jupyter Lab on port {port}")
        cmd = f"nohup jupyter lab --port {port} >/dev/null"
        print(cmd)
        run_cmd(cmd, verbose=False, async_cmd=True)
        print("=> Jupyter Lab launched")
        cls.tunnel_remote(port_dict={port: port}, remote_node=remote_node)

    @classmethod
    def tunnel_remote(cls, port_dict, remote_node):
        # forward remote ports to local
        hostname = cls._get_hostname()
        print(
            f"=> Forwarding ports from {remote_node} to {hostname}: {''.join([f'{k}:{v}' for k, v in port_dict.items()])}"
        )
        args = " ".join([f"-R {k}:localhost:{v}" for k, v in port_dict.items()])

        cmd = f"ssh {args} {remote_node}"
        # print(cmd)
        run_cmd(cmd, verbose=False, async_cmd=True)
        print("=> Remote Tunnel established")

    @classmethod
    def tunnel_local(cls, port_dict, remote_node):
        # forward local ports to remote
        hostname = cls._get_hostname()
        print(
            f"=> Forwarding ports from {hostname} to {remote_node}: {''.join([f'{k}:{v}' for k, v in port_dict.items()])}"
        )

        args = " ".join([f"-L {k}:localhost:{v}" for k, v in port_dict.items()])
        cmd = f"screen ssh {args} {remote_node}"
        # print(cmd)
        run_cmd(cmd, verbose=False)
        print("=> Local Tunnel Established")
