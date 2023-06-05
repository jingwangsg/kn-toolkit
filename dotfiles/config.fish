if status is-interactive
    # Commands to run in interactive sessions can go here
end

if test -f /export/home/kningtg/miniconda3/bin/conda
    status is-interactive && eval $HOME/miniconda3/bin/conda "shell.fish" "hook" $argv | source
end

abbr sv "~/server_utils/server.sh"
abbr lsq "python ~/server_utils/list_task.py"
abbr nv 'nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv'
abbr nvp "py3smi -f --left -w (math (tput cols)-20)"
# pip install py3nvml
abbr tc 'conda activate torch'
abbr zh 'conda activate zh'
abbr pyipdb "python -m ipdb -c continue "
abbr pypdb "python -m pdb -c continue "
# abbr aner "conda activate decouplener"
abbr fgA "python ~/server_utils/query_cluster.py --task available -n -1 | sort -n"
abbr knrsync "rsync -avur --progress --delete"
abbr bash_only "export NO_FISH=1; bash"

abbr gk "gpukill"
abbr g "gpu"
abbr dskill "knkill; and knkill deepspeed; and knkill acclerate"
abbr 'git?' 'copilot_git-assist'
abbr '??' 'copilot_what-the-shell'

function gpu
    export CUDA_VISIBLE_DEVICES=$argv
end

function rl
    readlink -f $argv
end

function fg 
    python ~/server_utils/query_cluster.py --task available $argv
end

function fu 
    python ~/server_utils/query_cluster.py --task usage $argv
end

function sg
  python $HOME/server_utils/dist_train.py --gpus $argv[1]
end

function knkill
    if test "$argv[1]" = "ALL"
        set argv
    end

    if test -z "$argv[1]"
        set filter "python"
    else
        set filter $argv[1]
    end

    ps -u kningtg --no-headers -o pid,comm= | grep -v -E "^\$|((string echo $PPID))|slurmstepd|bash|tmux" | grep -- $filter | awk '{print $1}' | xargs kill -9
end

function gpukill
    set variable $argv[1]
    for i in (string split "," $variable)
        nvidia-smi pmon -c 1 -d 1 -i $i | tail -n +3 | awk '{print $2}' | xargs kill -9
    end
end

function copilot_what-the-shell
    set TMPFILE (mktemp)
    function cleanup --on-event fish_exit; rm -f $copilot_what_the_shell_TMPFILE; end
    if github-copilot-cli what-the-shell $argv --shellout $TMPFILE
        if test -e "$TMPFILE"
            set FIXED_CMD (cat $TMPFILE)
            eval $FIXED_CMD
        else
            echo "Apologies! Extracting command failed"
        end
    else
        return 1
    end
end

function copilot_git-assist
    set TMPFILE (mktemp)
    function cleanup --on-event fish_exit; rm -f $copilot_what_the_shell_TMPFILE; end
    if github-copilot-cli git-assist $argv --shellout $TMPFILE
        if test -e "$TMPFILE"
            set FIXED_CMD (cat $TMPFILE)
            eval $FIXED_CMD
        else
            echo "Apologies! Extracting command failed"
        end
    else
        return 1
    end
end

function pcmd
  ps -p $argv[1] -o pid,ppid,cmd
end