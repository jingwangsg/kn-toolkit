# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
    *) return;;
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
    xterm-color|*-256color) color_prompt=yes;;
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
    xterm*|rxvt*)
        PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
    *)
    ;;
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

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
    if [ -f /usr/share/bash-completion/bash_completion ]; then
        . /usr/share/bash-completion/bash_completion
        elif [ -f /etc/bash_completion ]; then
        . /etc/bash_completion
    fi
fi
module load slurm

alias sv="conda activate torch && ~/server_utils/server.sh"
alias lsq="python ~/server_utils/list_task.py"
alias fg="conda activate torch && python ~/server_utils/query_cluster.py --task available"
alias fu="conda activate torch && python ~/server_utils/query_cluster.py --task usage"
alias nv='nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv'
alias nvp="conda activate torch && nvidia-htop.py -c"
alias tc='conda activate torch'
alias pyipdb="python -m ipdb -c continue "
alias pypdb="python -m pdb -c continue "
# alias aner="conda activate decouplener"
alias tf2="conda activate tf2"
alias fgA="conda activate torch && python ~/server_utils/query_cluster.py --task available -n -1| sort -n"
alias sg="python $HOME/server_utils/dist_train.py --gpus"
alias gk="gpukill"
alias g="gpu"
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

gpukill() {
    variable=$1
    for i in ${variable//,/ }
    do
        nvidia-smi pmon -c 1 -d 1 -i $i | tail -n +3 | awk '{print $2}' | xargs kill -9
    done
}
rl() {
    readlink -f $1
}

# export PS1="[\u@\h \W]\n\$"
export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\n\$"
export WANDB_DIR="$HOME/.wandb"
export SLURM_TMPDIR="$HOME/.tmp"
export TMUX_TMPDIR="$HOME/.tmp"

show_gpu() {
    echo $CUDA_VISIBLE_DEVICES
}

gpu() {
    export CUDA_VISIBLE_DEVICES="$1"
}
# https://stackoverflow.com/questions/58707855/how-to-use-alias-to-simplify-cuda-visible-devices
cuda () {
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

ROOTDIR=$HOME/app
export LD_LIBRARY_PATH="/cm/shared/apps/cudnn8.4.1/usr/lib/x86_64-linux-gnu:$ROOTDIR/lib:$ROOTDIR/lib64"
export CUDA_TOOLKIT_ROOT="/cm/shared/apps/cuda11.6/toolkit/11.6.0"
# export CUDA_HOME="/cm/local/apps/cuda/libs/current/"
export CUDA_HOME="/cm/shared/apps/cuda11.6/toolkit/11.6.0"
PATH="$ROOTDIR/bin:$ROOTDIR/include:$ROOTDIR/lib:$ROOTDIR/lib/pkgconfig:$ROOTDIR/lib/share/pkgconfig:$ROOTDIR/lib64/:$PATH"
PATH="$CUDA_TOOLKIT_ROOT/:$CUDA_HOME:$PATH"
export PATH="$CUDA_HOME/bin/:$CUDA_TOOLKIT_ROOT/bin/:$PATH"
LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
# LIBRARY_PATH="$CUDA_HOME/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64:$LD_LIBRARY_PATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
