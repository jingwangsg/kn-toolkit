import argparse
from datetime import datetime, timedelta, timezone

from ..utils.mail import send_email
from ..utils.system import run_cmd


def main():
    hostname = run_cmd("hostname").stdout.strip()
    DEFAULT_BODY = [
        "Hostname: " + hostname,
    ]

    nvidia_output = run_cmd("nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader")
    if nvidia_output.returncode != 0:
        DEFAULT_BODY += [
            "GPU not found",
        ]
    else:
        DEFAULT_BODY += [
            "GPU Information:",
            nvidia_output.stdout,
        ]

    utc_dt = datetime.now(timezone.utc)  # UTC time
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))  # Beijing time
    DEFAULT_BODY += [
        "Time: " + bj_dt.strftime("%Y-%m-%d %H:%M:%S"),
    ]

    DEFAULT_BODY = ["---------------Meta Information----------------"] + DEFAULT_BODY
    DEFAULT_BODY += ["----------------------------------------------------"]

    parser = argparse.ArgumentParser(description="Send email")
    parser.add_argument("--to", type=str, help="To email address")
    parser.add_argument("--subject", type=str, help="Subject of the email", default="Message from GPU Cluster")
    parser.add_argument("-m", "--message", help="Body of the email", nargs="+", default=[], type=str)
    args = parser.parse_args()

    message = DEFAULT_BODY + args.message

    send_email(
        to_addr="kningtg@gmail.com",
        subject=f"({hostname})" + args.subject,
        text="\n".join(message),
    )
