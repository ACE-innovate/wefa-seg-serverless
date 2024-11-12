# Use the specified PyTorch image with CUDA 12.1 and cuDNN 9
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install dependencies for Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p /opt/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm /opt/miniconda3/miniconda.sh

# Set environment variables for Conda
ENV PATH /opt/miniconda3/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

WORKDIR /opt

RUN git clone https://github.com/ACE-innovate/wefa-seg-serverless

# create the Conda environment
RUN conda env create -f /opt/wefa-seg-serverless/anydoor/environment.yaml

# Set up the shell to use the Conda environment by default
SHELL ["conda", "run", "-n", "anydoor", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
