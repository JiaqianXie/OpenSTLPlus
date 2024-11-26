FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV WANDB_API_KEY=b118bf8a81ef06fe0c1e2a6324fb81a510fc608c

COPY ./openstl/ /app/openstl
COPY ./setup.py /app/setup.py
COPY ./tools/ /app/tools
COPY ./requirements/ /app/requirements

# Install required system libraries
# Note: ffmpeg adds 0.38GB to image size, but likely that we could just depend on minimal dependencies if we had time to work out what we specifically need.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            git ffmpeg wget tzdata python3.8 python3-pip python3.8-venv python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install packaging
RUN pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install mamba-ssm[causal-conv1d]
RUN pip3 install lightning xarray dask netCDF4 bottleneck tensorboard
RUN pip3 install timm==0.6.11
RUN pip3 install scikit-image hickle decord fvcore lpips nni einops pandas tqdm wandb dill optuna
RUN pip3 install -e .

ENTRYPOINT ["tail", "-f", "/dev/null"]
