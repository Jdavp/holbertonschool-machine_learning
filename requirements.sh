#!/usr/bin/env bash
#update and installs
sudo apt-get update
sudo apt-get install python3-tk
#installing pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
#was succesfully dowload
pip -V
#modules
pip install --user numpy==1.15
pip install --user scipy==1.3
pip install --user pycodestyle==2.5
pip install --user matplotlib
pip install --user pillow
#pep8
sudo pip install pep8
sudo pip install --upgrade pep8
sudo pip uninstall pep8
#  installing tensorflow
pip install --user tensorflow==1.12
#optimize Tensorflow
# install all dependencies
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python python3-dev
# install Bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh
chmod +x bazel-0.18.1-installer-linux-x86_64.sh
sudo ./bazel-0.18.1-installer-linux-x86_64.sh --bin=/bin
#clone Tensorflow repo
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.12
#build and install Tensorflow
export PYTHON_BIN_PATH=/usr/bin/python3 # or wherever python3 is located
bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install --user /tmp/tensorflow_pkg/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl
#remove Tensorflow Repo
cd ..
rm -rf tensorflow
