#!/bin/bash
cmd=$(python ~/server_utils/server_helper.py --server_nos $1)

module load cuda11.2/toolkit/11.2.0
echo $cmd
$cmd
