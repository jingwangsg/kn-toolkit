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
    print(lines)
    if lines == "":
        return 
    
    lines = lines.split("\n")
    for line in lines:
        pid = line.split()[0]
        if pid == str(os.getpid()):
            print("skipping myself")
            continue
        print(f"killing {pid}")
        subprocess.run(f"kill -9 {pid}", shell=True)

if __name__ == "__main__":
    args = parse_args()
    kill(args)
