#!/bin/bash

echo_and_eval() {
    echo "$@"
    eval "$@"
}

echo "==================install miniconda=================="
mkdir -p $HOME/Downloads
cd $HOME/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

echo "==================setup env=================="
conda init bash
source $HOME/.bashrc
conda create -n torch python=3.10 -y
conda activate torch
echo_and_eval "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"

echo "==================install berw=================="
cd $HOME
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install fish -vd

echo "==================install vscode cli=================="
cd $HOME/Downloads
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xvf vscode_cli.tar.gz
mv code $HOME/