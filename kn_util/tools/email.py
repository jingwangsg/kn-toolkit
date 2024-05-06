import argparse
from ..utils.system import run_cmd
from ..utils.mail import send_email


def main():
    hostname = run_cmd("hostname").stdout.strip()
    DEFAULT_BODY = [
        "Hostname: " + hostname,
    ]

    nvidia_output = run_cmd("nvidia-smi")
    if nvidia_output.returncode != 0:
        DEFAULT_BODY += [
            "GPU not found",
        ]
    else:
        DEFAULT_BODY += [
            "GPU Information:",
            nvidia_output.stdout,
        ]

    DEFAULT_BODY =  ["---------------Meta Information----------------"] + DEFAULT_BODY
    DEFAULT_BODY += ["----------------------------------------------------"]

    parser = argparse.ArgumentParser(description="Send email")
    parser.add_argument("--to", type=str, help="To email address")
    parser.add_argument("--subject", type=str, help="Subject of the email", default=f"Message from GPU Cluster ({hostname})")
    parser.add_argument("-m", "--message", help="Body of the email", nargs="+", default=[], type=str)
    args = parser.parse_args()

    message = DEFAULT_BODY + args.message

    send_email(
        to_addr="kningtg@gmail.com",
        subject=args.subject,
        text="\n".join(message),
    )
