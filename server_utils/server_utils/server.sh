#!/bin/bash
cmd=$(python ~/server_utils/server_helper.py --server_nos $1)

# module load cuda90/toolkit/9.0.176
echo $cmd
$cmd
