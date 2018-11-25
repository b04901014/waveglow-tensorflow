# ----------------------------------------------------------------------------
# Add cuDNN to nvidia/cuda:9.2-devel
# ----------------------------------------------------------------------------
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# ----------------------------------------------------------------------------
# Install Required Packages
# ----------------------------------------------------------------------------
RUN apt-get update
RUN apt-get install -y software-properties-common \
                       vim \
                       git-core \
                       aptitude \
                       portaudio19-dev \
                       python3-tk \
                       tmux \
                       psmisc \
                       libsndfile-dev \
                       libssl-dev \
                       curl

RUN alias python=python3 && \
    alias pip=pip3 && \
    ln -s /usr/bin/python3.5 /usr/bin/python && \
    echo 'alias python=python3' >> ~/.bashrc && \
    echo 'alias pip=pip3' >> ~/.bashrc

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# ----------------------------------------------------------------------------
# Install Tensorflow from source
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-dev-9-2 \
        cuda-cudart-dev-9-2 \
        cuda-cufft-dev-9-2 \
        cuda-curand-dev-9-2 \
        cuda-cusolver-dev-9-2 \
        cuda-cusparse-dev-9-2 \
        git \
        libnccl2=2.2.13-1+cuda9.2 \
        libnccl-dev=2.2.13-1+cuda9.2 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        libpython3.5-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.2/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN apt-get update && \
        apt-get install -y nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.2 && \
        apt-get update && \
        apt-get install -y libnvinfer4=4.1.2-1+cuda9.2 && \
        apt-get install -y libnvinfer-dev=4.1.2-1+cuda9.2

# Link NCCL libray and header where the build script expects them.
RUN mkdir /usr/local/cuda-9.2/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h


RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        mock \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
    python3 -m ipykernel.kernelspec

# RUN ln -s -f /usr/bin/python3 /usr/bin/python#

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.15.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /tensorflow
RUN git clone --branch=r1.12 --depth=1 https://github.com/tensorflow/tensorflow.git .

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python3
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=$CUDA_VERSION
ENV TF_CUDNN_VERSION=$CUDNN_VERSION

# NCCL 2.x
ENV TF_NCCL_VERSION=2

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
	--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

WORKDIR /root/

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888


# ----------------------------------------------------------------------------
# Configure locales
# ----------------------------------------------------------------------------

RUN aptitude install locales && \
    dpkg-reconfigure locales && \
    echo "Asia/Hong Kong" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    echo 'LANG="en_US.UTF-8"'>/etc/default/locale && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

# ----------------------------------------------------------------------------
# Modify bashrc file, configure vimrc file
# ----------------------------------------------------------------------------

RUN echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc && \
    echo 'export LANG=en_US.UTF-8' >> ~/.bashrc && \
    echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bashrc && \
    echo 'nnoremap <C-Left> :tabprevious<CR>' >> ~/.vimrc && \
    echo 'nnoremap <C-Right> :tabnext<CR>' >> ~/.vimrc && \
    echo 'set expandtab' >> ~/.vimrc && \
    echo 'set shiftwidth=2' >> ~/.vimrc


# ----------------------------------------------------------------------------
# Other Python dependencies
# ----------------------------------------------------------------------------

COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt

WORKDIR /root/
CMD ["sleep", "infinity"]
