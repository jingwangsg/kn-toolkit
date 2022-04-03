ret_str=$(sed -n "$1p" ~/server_utils/server_list.txt)

IFS=','
read -ra stdarr <<< "$ret_str"

echo $1,${stdarr[1]},${stdarr[2]}

module load cuda90/toolkit/9.0.176
srun -p ${stdarr[1]} -w node$1 --export ALL --mem=0 --exclusive --pty bash 