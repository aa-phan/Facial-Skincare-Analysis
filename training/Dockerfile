# Use a lightweight Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Set the NVIDIA runtime environment variable
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Update Conda and create the environment with Python 3.9
RUN conda update -n base -c defaults conda && \
conda create -n skingpt4_llama2 python=3.9 && \
conda clean -a

# Activate the environment and install packages with CUDA 12.5
RUN /bin/bash -c "source activate skingpt4_llama2 && \
conda install -c conda-forge mamba=1.4.7 && \
pip3 install torch torchvision torchaudio pytest"


RUN useradd -m dockeruser
USER dockeruser

# Set the default shell to activate the conda environment
SHELL ["conda", "run", "-n", "skingpt4_llama2", "/bin/bash", "-c"]

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
