import socket
import torch.distributed as dist
import os
import datetime
import torch

def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]

def initialize_ddp_from_env():
    # local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=rank,
                            timeout=datetime.timedelta(seconds=5400))

    torch.cuda.set_device(local_rank)