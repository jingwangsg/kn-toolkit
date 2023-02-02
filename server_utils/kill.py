import subprocess
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    return parser.parse_args()

def kill_(args):

    lines = subprocess.run(
        f"ps -u kningtg -o pid,command | awk '\{print \$2\}' | grep python", capture_output=True, text=True, shell=True
    ).stdout.strip()
    if lines == "":
        return 
    
    lines = lines.split("\n")
    for line in lines:
        pid = line.split()[1]
        if str(os.getpid()) == pid:
            continue
        subprocess.run(f"kill -9 {pid}", shell=True)
        print(line)

def kill(args):
    subprocess.run(f"")

if __name__ == "__main__":
    args = parse_args()
    kill(args)
