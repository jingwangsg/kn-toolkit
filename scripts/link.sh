#/bin/bash

echo "=> create symlink for .bashrc"
rm -rf ~/.bashrc
ln -s $(pwd)/dotfiles/.bashrc ~/.bashrc

echo "=> create symlink for server_utils"
rm -rf ~/server_utils
ln -s $(pwd)/server_utils ~/server_utils
echo $(ls -alF ~/server_utils)

echo "=> create symlink for .tmux.conf"
rm -rf ~/.tmux.conf
ln -s $(pwd)/dotfiles/.tmux.conf ~/.tmux.conf
echo $(ls -alF ~/.tmux.conf)

echo "=> create symlink for config.fish"
rm -rf ~/.config/fish/config.fish
ln -s $(pwd)/dotfiles/config.fish ~/.config/fish/config.fish
echo $(ls -alF ~/.config/fish/config.fish)

echo "=> create symlink for grab_gpu"
mkdir -p $HOME/miniconda3/envs/cuda$CUDA_VERSION/bin/
rm -rf $HOME/miniconda3/envs/cuda$CUDA_VERSION/bin/python
ln -s $(pwd)/server_utils/bin/python.cu$CUDA_VERSION $HOME/miniconda3/envs/cuda$CUDA_VERSION/bin/python
echo $(ls -alF $HOME/miniconda3/envs/cuda$CUDA_VERSION/bin/python)

echo "=> create symlink for kn_util"
TORCH_ENV_PATH=$(ls -d $HOME/miniconda3/envs/torch/lib/*/ | grep python | tail -1)
if [ -e "$HOME/miniconda3/envs/torch" ]
then
    if [ -e "$TORCH_ENV_PATH/site-packages/kn_util" ]
    then
        echo "kn_util already exists"
    else
        ln -s $(pwd)/kn_util $TORCH_ENV_PATH/site-packages/kn_util
    fi
    echo $(ls -alF $TORCH_ENV_PATH/site-packages/kn_util)
else
    echo "conda torch env not exists yet"
fi