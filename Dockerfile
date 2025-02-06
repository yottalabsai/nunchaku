FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget 

ENV CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $CONDA_URL -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    source ~/.bashrc && \
    conda init

RUN conda create -n image python=3.12 && \
    conda activate image && \
    pip install torch torchvision torchaudio && \
    pip install diffusers ninja wheel transformers accelerate sentencepiece protobuf && \
    pip install huggingface_hub peft opencv-python einops gradio spaces GPUtil && \
    conda install -c conda-forge gxx=11 gcc=11


ENV MAX_JOBS=4

COPY . /app

RUN git submodule init && \
    git submodule update && \
    python3 setup.py develop

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]