import subprocess
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    format = "USER@IP:DIR"
    args.add_argument("from_dir", type=str, help=format)
    args.add_argument("to_dir", type=str, help=format)

    return args.parse_args()

def parse(s):
    user = None
    ip = None
    dir_path = None
    is_remote = None
    
    if len(s.split("@")) == 2:
        user, _ = s.split("@")
        ip, dir_path = s.split(":")
        is_remote = True
    else:
        dir_path = s
        is_remote = False

    return user, ip, dir_path, is_remote

def main(args):
    from_dir = args.from_dir
    to_dir = args.to_dir
    from_user, from_ip, from_path, from_is_remote = parse(from_dir)
    to_user, to_ip, to_path, to_is_remote = parse(to_dir)

    if from_is_remote and to_is_remote:
        print("using remote - remote")
        cmd = f"ssh -R 127.0.0.1:9999:{to_ip} {from_user}@${from_ip} 'rsync -au -avz -v -P -e \"ssh -p 9999\" {from_path} {to_user}@127.0.0.1:{to_path}'"
    else:
        print("using local - remote")
        cmd = f"rsync -avzP {from_dir} {to_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
