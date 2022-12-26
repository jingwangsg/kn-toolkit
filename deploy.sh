rm -rf ~/.bashrc
# cp .bashrc ~/.bashrc
ln -s $(pwd)/.bashrc ~/.bashrc
rm -rf ~/server_utils
ln -s $(pwd)/server_utils ~/server_utils