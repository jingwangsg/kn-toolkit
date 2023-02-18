#!/bin/bash

source ~/.bashrc

no=$1
shift
prefix=$(python ~/server_utils/server_helper.py --server_nos $no)
default='--pty bash'
cmd="$prefix ${@:-$default}"

echo $cmd
$cmd
