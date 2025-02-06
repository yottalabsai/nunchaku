FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN echo $SHELL
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget 



RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash && \
    source .bashrc

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