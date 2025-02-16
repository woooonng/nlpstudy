FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git && apt-get install -y tmux

RUN pip install --upgrade pip

# 필요한 Python 패키지 설치
RUN pip install -U transformers>=4.48.0
RUN pip install datasets
RUN pip install tqdm
RUN pip install omegaconf
RUN pip install wandb

WORKDIR /

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME