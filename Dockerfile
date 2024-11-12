# Use the specified PyTorch image with CUDA 12.1 and cuDNN 9
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install dependencies for Miniconda
RUN apt-get update && apt-get install -y wget && apt-get install -y git  && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Set your GitHub token as an ARG
ARG GITHUB_TOKEN

# Use the token to clone the repository
RUN git clone https://$GITHUB_TOKEN@github.com/ACE-innovate/wefa-seg-serverless.git /opt/wefa-seg-serverless

RUN pip install -r /opt/wefa-seg-serverless/requirements.txt

# Default command
CMD ["bash", "-c", "conda activate anydoor && python /opt/wefa-seg-serverless/anydoor/runpod_handler.py"]