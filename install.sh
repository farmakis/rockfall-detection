#!/bin/bash

# Local variables
PROJECT_NAME=3d-ml-rd
PYTHON=3.6
TF=2.2.0
CUDA_SUPPORTED=(10.1)

# Recover the project's directory from the position of the install.sh
# script and move there. Not doing so would install some dependencies in
# the wrong place
HERE=`dirname $0`
HERE=`realpath $HERE`
cd $HERE

echo
echo "⭐ Searching for installed CUDA"
echo
# Recover the CUDA version using nvcc
CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`
CUDA_MAJOR=`echo ${CUDA_VERSION} | sed 's/\..*//'`
CUDA_MINOR=`echo ${CUDA_VERSION} | sed 's/.*\.//'`

# Recover cuda root
if [ -L /usr/local/cuda ]; then
    CUDA_ROOT=$(readlink -f /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR})
elif command -v nvcc &> /dev/null; then
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
else
    echo "CUDA not found"
    exit 1
fi

# If CUDA version not supported, print error and exit
if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "Found CUDA ${CUDA_VERSION} installed, which is not among the supported versions: "`echo ${CUDA_SUPPORTED[*]}`
    echo "Please update CUDA to one of the supported versions."
    exit 1
fi

echo
echo
echo "⭐ Searching for installed conda"
echo
# Recover the path to conda on your machine
# First search the default '~/miniconda3' and '~/anaconda3' paths. If
# those do not exist, ask for user input
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide your conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh

echo
echo
echo "⭐ Creating conda environment '${PROJECT_NAME}'"
echo
# Create deep_view_aggregation environment from yml
conda create --name ${PROJECT_NAME} python=${PYTHON} -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

echo
echo
echo "⭐ Compiling custom TensorFlow ops"
echo
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

g++ -std=c++11 -shared ./3d_interpolation/tf_interpolate.cpp -o ./3d_interpolation/tf_interpolate_so.so  -I $CUDA_ROOT/include -lcudart -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./grouping/tf_grouping_g.cu -o ./grouping/tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./grouping/tf_grouping.cpp ./grouping/tf_grouping_g.cu.o -o ./grouping/tf_grouping_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./sampling/tf_sampling_g.cu -o ./sampling/tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./sampling/tf_sampling.cpp ./sampling/tf_sampling_g.cu.o -o ./sampling/tf_sampling_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2


echo
echo
echo "⭐ Installing conda and pip dependencies"
echo
conda install pip nb_conda_kernels -y
pip install tensorflow-gpu==2.2.0
pip install tensorboard
pip install pyrootutils
pip install hydra-core --upgrade
pip install hydra-colorlog
pip install open3d==0.13.0
pip install trimesh
pip install sklearn
pip install progress

# let user know
echo
echo
echo "🚀 Successfully installed 3DML-RD"