# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
*i*) ;;
*) return ;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
xterm-color | *-256color) color_prompt=yes ;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm* | rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*) ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

source /etc/profile.d/modules.sh
LATEST_CUDA_MODULE=$(module avail | grep 'cuda' | tr ' ' '\n' | grep -v '^$' | grep 'cuda11.*/toolkit' | sort -n | tail -n 1)
module load slurm cuda11.7/toolkit/11.7.1

alias sv="conda activate torch && ~/server_utils/server.sh"
alias lsq="python ~/server_utils/list_task.py"
alias fg="conda activate torch && python ~/server_utils/query_cluster.py --task available"
alias fu="conda activate torch && python ~/server_utils/query_cluster.py --task usage"
alias nv='nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv'
# alias nvp="gpustat -f"
alias nvp="py3smi -f --left -w $(($(tput cols) - 20))"
# pip install py3nvml
alias tc='conda activate torch'
alias pyipdb="python -m ipdb -c continue "
alias pypdb="python -m pdb -c continue "
# alias aner="conda activate decouplener"
alias tf2="conda activate tf2"
alias fgA="conda activate torch && python ~/server_utils/query_cluster.py --task available -n -1| sort -n"
alias sg="python $HOME/server_utils/dist_train.py --gpus"

alias gk="gpukill"
alias g="gpu"
alias kps="ps --no-header -eo ppid,user,stime,cmd | sort -u -k1,1 | cut -c 1-$(($(tput cols) - 50)) | grep -viE 'root|postfix'"

knkill() {
    # conda activate torch
    # python $HOME/server_utils/kill.py ${1:-python}
    ps -u kningtg -o pid,command | awk '{print $1,$2}' | grep ${1:-python} | awk '{print $1}' | xargs kill -9
}

upl() {
    commit_content=${1:-tmp commit $(date)}
    git add .
    git commit -m "$commit_content"
    git push
}

pcmd() {
    ps -p $1 -o command | tail -n +2 | fold -w $(($(tput cols) - 20))
}

gpukill() {
    variable=$1
    for i in ${variable//,/ }; do
        nvidia-smi pmon -c 1 -d 1 -i $i | tail -n +3 | awk '{print $2}' | xargs kill -9
    done
}
rl() {
    readlink -f $1
}

# export PS1="[\u@\h \W]\n\$"
export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\n\$"
export WANDB_DIR="$HOME/.wandb"

if [ ! -d "$HOME/.tmp/$(hostname)" ]; then
    mkdir -p "$HOME/.tmp/$(hostname)"
fi
export SLURM_TMPDIR="$HOME/.tmp/$(hostname)"
export TMUX_TMPDIR="$HOME/.tmp/$(hostname)"
export TMPDIR="$HOME/.tmp/$(hostname)"
export HOMEBREW_TEMP="$HOME/.tmp/$(hostname)"

show_gpu() {
    echo $CUDA_VISIBLE_DEVICES
}

gpu() {
    export CUDA_VISIBLE_DEVICES="$1"
}
# https://stackoverflow.com/questions/58707855/how-to-use-alias-to-simplify-cuda-visible-devices
cuda() {
    local devs=$1
    shift
    CUDA_VISIBLE_DEVICES="$devs" "$@"
}
knrun() {
    local devices=$1
    shift
    local nproc_per_node=$(($(echo $devices | grep -o "," | wc -l) + 1))
    CUDA_VISIBLE_DEVICES=$devices torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc_per_node $@
}
knrun_mn() {
    local nnodes=$1
    shift
    local node_rank=$1
    shift
    local ngpu=$1
    shift
    local port=$1
    shift
    torchrun --rdzv_backend=c10d --rdzv_endpoint=155.69.144.22:$port --nnodes=$nnodes --node_rank=$node_rank $node--nproc_per_node=$ngpu $@
}
ca() {
    conda activate "$1"
}
kill_nfs() {
    # 递归搜索当前文件夹内所有 .nfs 文件
    nfs_files=$(find . -type f -name '*.nfs')

    # 遍历所有 .nfs 文件
    for nfs_file in $nfs_files; do
        # 使用 lsof 命令找到正在使用这个文件的进程
        processes=$(lsof "$nfs_file" | awk 'NR>1 {print $2}' | sort -u)

        # 终止这些进程
        for process in $processes; do
            echo "Killing process: $process"
            kill -9 "$process"
        done
    done

}

ROOTDIR=$HOME/usr
HOMEBREW=$HOME/homebrew
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH:$HOMEBREW/lib
export HOMEBREW_NO_INSTALL_FROM_API=1
export LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT=$(which nvcc | sed 's/\/bin\/nvcc//g')
PATH="$HOME/miniconda3/bin:$PATH"
PATH="$HOMEBREW/bin:$PATH"
# PATH="$CUDA_TOOLKIT_ROOT/:$PATH"
# PATH="$CUDA_HOME/bin/:$CUDA_TOOLKIT_ROOT/bin/:$PATH"
export PATH
if [ -z $NO_FISH ]; then
    exec fish
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/export/home2/kningtg/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/export/home2/kningtg/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/export/home2/kningtg/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/export/home2/kningtg/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
