#!/bin/bash
set -ex

STAGE=${1:-"clone"}

if [ "$STAGE" == "clone" ]; then

    if [ ! -d ~/homebrew ]; then
        git clone https://github.com/Homebrew/brew homebrew
    fi

    # create link only when python39 does not exist
    if [ ! -f ~/homebrew/bin/python39 ]; then
        ln -s /bin/bash ~/homebrew/bin/python39
    fi

    brew update --force --quiet
fi

if [ $STAGE == "install" ]; then
    # hostname should not be started with scse (e.g. scsehg)
    cur_hostname=$(hostname)
    if [[ $cur_hostname == scse* ]]; then
        echo "You are installing in master node, this may cause compatibility issues."
        echo "Please install in a non-master node."
        exit 1
    fi

    brew install --build-from-source glibc

    brew install patchelf

    python -m kn_util.tools.brew patch --app patchelf -y

    brew install rclone
    if [ ! -f ~/homebrew/bin/fusermount3 ]; then
        ln -s /bin/fusermount ~/homebrew/bin/fusermount3
    fi

    python -m kn_util.tools.brew patch -y

    brew install btop
    brew install ncdu
    brew install -f gdu
    brew install tmux
    python -m kn_util.tools.brew patch -y

    brew install git
    brew install git-lfs
    brew install fd
    brew install fzf
    brew install tree

    python -m kn_util.tools.brew patch -y

    brew install fish
    python -m kn_util.tools.brew patch -y
    if [ ! -f ~/homebrew/bin/python311 ]; then
        ln -s ~/homebrew/bin/fish ~/homebrew/bin/python311
    fi

    # brew install ffmpeg
    python -m kn_util.tools.brew install ffmpeg --post_patch
    python -m kn_util.tools.brew patch -y

fi
