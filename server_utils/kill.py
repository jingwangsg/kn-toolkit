import subprocess
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    return parser.parse_args()

def kill(args):
    lines = subprocess.run(
        f"ps -u kningtg | grep {args.type}", capture_output=True, text=True, shell=True
    ).stdout.strip()
    if lines == "":
        return 
    
    lines = lines.split("\n")
    for line in lines:
        pid = line.split()[0]
        if str(os.getpid()) == pid:
            continue
        subprocess.run(f"kill -9 {pid}", shell=True)
        print(line)

if __name__ == "__main__":
    args = parse_args()
    kill(args)
