from ast import arg
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
import ipdb
import torch
import time
import argparse
import re
import os
from tqdm import tqdm
import subprocess
import pandas as pd
import numpy as np
import smtplib

device = None

import cmd

def read_args():
    global M10
    args = argparse.ArgumentParser()
    args.add_argument("--interval", type=float, default=0.1)
    # args.add_argument("--tensor_size", type=int, default=1, help="x 1000M")
    args.add_argument("-m", "--mem_lim", type=float, default=np.inf, help="(G)")
    args.add_argument("-g", "--gpu_id", type=int, default=0)
    args.add_argument("-s", "--send_email", action="store_true")
    return args.parse_args()

def get_gpu_info(gpu_id):
    cmd = """
    nvidia-smi --query-gpu=gpu_name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used\
    --format=csv,noheader,nounits
    """

    ret_arr = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.split("\n")[gpu_id].split(",")
    if ret_arr[0] == '': return None
    parsed_ret_arr = []
    for x in ret_arr:
        try:
            parsed_ret_arr.append(eval(x))
        except:
            parsed_ret_arr.append(x)

    columns = ["name", "utils", "util_mem", "mem_total", "mem_free", "mem_used"]
    info_dict = dict(zip(columns, parsed_ret_arr))

    return info_dict

def get_pid_info():
    global device
    pid = os.getpid()
    cmd = f"nvidia-smi | grep '{pid}      C'"
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    captured_mem = int(re.search("([0-9]*)MiB", ret).group(1))

    return captured_mem


def send_email(args):
    username = "kningtg@163.com"
    password = "RQVJCZRHGLTULJEJ"
    from_addr = "kningtg@163.com"
    to_addr = "knjingwang@gmail.com"
    hostname = subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout
    gpu_id = args.gpu_id
    gpu_info = get_gpu_info(gpu_id)

    subject = f"[SG] {hostname} GPU#{gpu_id}:{gpu_info['name']}"
    text = pd.DataFrame.from_dict(gpu_info, orient="index").to_string(header=False) + f"\nCaptured Memory: {get_pid_info()} Mb"

    message = MIMEText(text, "plain", "utf-8")
    message["From"] = formataddr((str(Header("GPU Cluster", "utf-8")), username))
    message["To"] = to_addr
    message["Subject"] = Header(subject, "utf-8")

    # print(message.as_string())

    smtp_server = smtplib.SMTP_SSL('smtp.163.com', 465)
    smtp_server.login(username, password)
    smtp_server.sendmail(from_addr, to_addr, message.as_string())
    smtp_server.close()
    print("=> Email sent!")


def scrumble_gpu():
    args = read_args()
    global M10, device

    gpu_id = args.gpu_id
    device = torch.device(f"cuda:{gpu_id}")
    interval = args.interval
    tensor_list = []
    
    gpu_info = get_gpu_info(args.gpu_id)
    hostname = subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout
    print(pd.DataFrame.from_dict(gpu_info, orient="index").to_string(header=False))

    mem_total = gpu_info["mem_total"]


    mem_lim = np.ceil(args.mem_lim * 1024)
    gpu_id = args.gpu_id

    # print(torch.cuda.)
    # import ipdb; ipdb.set_trace()
    left_mem = 256
    mem_lim = min(mem_total - left_mem, mem_lim)
    disp_interval = 1000
    while True:
        try:
            torch.randn(1).to(device)
            break
        except:
            pass
    
    cnt = 0
    with tqdm(total=mem_lim, desc=f"gpu_{gpu_id}") as pbar:

        captured_mem = get_pid_info()
        pbar.n = captured_mem
        pbar.update(0)
        
        while captured_mem <= mem_lim:
            mem_free = get_gpu_info(gpu_id)["mem_free"]
            tensor_size = 26315790
            if mem_free > 3000:
                tensor_size = (get_gpu_info(gpu_id)["mem_free"] // 1000 - 1) * 263157900
            if mem_free < 100:
                tensor_size = 10000
            
            try:
                tensor_list.append(torch.randn(tensor_size).to(device))
                captured_mem = get_pid_info()
                pbar.n = captured_mem
                pbar.update(0)
                cnt += 1
            except:
                time.sleep(interval)
            
        
    del tensor_list
    # tensor_list.append(torch.randn(2631579*1200).to(device))
    print("-" * 10 + f" Capture Memory: {captured_mem:7d} Mb " +"-" * 10)
    if args.send_email:
        send_email(args)

    mode = input("shell/python:")

    if mode == "s":
        while True:
            cmd = input("$")
            if cmd == "EOF" or cmd == "q":
                quit()
            else:
                subprocess.run(cmd, shell=True)
    else:
        ipdb.set_trace()

if __name__ == "__main__":
    scrumble_gpu()