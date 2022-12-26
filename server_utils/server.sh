#!/bin/bash
cmd=$(python ~/server_utils/server_helper.py --server_nos $1)

echo $cmd
$cmd
