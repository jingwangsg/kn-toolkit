import argparse
from ..utils.system import run_cmd

parser = argparse.ArgumentParser(description='Send an email')
parser.add_argument("content", help="The content of the email")
args = parser.parse_args()

from ..utils.mail import send_email

hostname = run_cmd("hostname").stdout.strip()
send_email(to_addr="kningtg@gmail.com", subject=f"System Notification {hostname}", text=args.content)

print("=> Email sent successfully")