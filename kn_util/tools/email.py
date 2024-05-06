import argparse
from ..utils.system import run_cmd
from ..utils.mail import send_email


def main():
    hostname = run_cmd("hostname").stdout.strip()
    DEFAULT_BODY = [
        "Hostname: " + hostname,
        "GPU Info: ",
        run_cmd("nvidia-smi").stdout,
    ]

    parser = argparse.ArgumentParser(description="Send email")
    parser.add_argument("--to", type=str, help="To email address")
    parser.add_argument("--subject", type=str, help="Subject of the email", default=f"Message from GPU Cluster ({hostname})")
    parser.add_argument("-m", "--message", type=str, help="Body of the email", default=None)
    args = parser.parse_args()

    if args.message is not None:
        DEFAULT_BODY.append(args.message)

    send_email(
        to_addr="kningtg@gmail.com",
        subject=args.subject,
        text="\n".join(DEFAULT_BODY),
    )
