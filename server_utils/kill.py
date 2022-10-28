import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    return parser.parse_args()

def kill(args):
    lines = subprocess.run(
        "ps -u kningtg | grep {args.type}", capture_output=True, text=True, shell=True
    ).stdout.strip()
    print(lines)
    if lines == "":
        return 
    
    lines = lines.split("\n")
    for line in lines:
        pid = line.split()[0]
        subprocess.run(f"kill -9 {pid}", shell=True)

if __name__ == "__main__":
    args = parse_args()
    kill(args)
