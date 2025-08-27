FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OS & build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates tini vim gosu \
    build-essential python3-dev pkg-config ninja-build \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ----- Miniconda -----
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ALWAYS_YES=true

# Non-root (better file perms on mounted volumes)
ARG USERNAME=appuser
ARG USERID=1000

RUN useradd -m -s /bin/bash -u ${USERID} -g 1000 -o ${USERNAME}

WORKDIR /workspace

# - environment.yml: CONDA-ONLY (no pip block)
# - requirements.txt: PIP packages (EXCEPT torch/torchvision)
COPY environment.yml* requirements.txt* ./

# conda-forge only; flexible priority to reduce conflicts
RUN printf "channels:\n  - conda-forge\nchannel_priority: flexible\n" > /opt/conda/.condarc

# mamba for faster, reliable solves
RUN conda install -n base -c conda-forge mamba && conda clean -afy

# 1) Create/Update base env from environment.yml (conda packages only)
RUN if [ -f "environment.yml" ]; then \
      mamba env update --name base --file environment.yml && \
      mamba clean --all -y ; \
    fi

# --- CUDA / Torch config for builds and runtime ---
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0+PTX;8.9;8.6" \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    MAX_JOBS=8

# 2) Install PyTorch (CUDA 12.4 wheels)  ← cu124 wheels work with host driver 12.8
ARG TORCH_VER=2.7.0
ARG TVISION_VER=0.22.0
RUN pip install --no-cache-dir \
    torch==${TORCH_VER} torchvision==${TVISION_VER} torchaudio==${TORCH_VER} \
    --index-url https://download.pytorch.org/whl/cu128



# 3) Install pip packages (from requirements.txt) — no torch/torchvision here
ENV PIP_NO_CACHE_DIR=1
RUN if [ -f "requirements.txt" ]; then \
      pip install -r requirements.txt ; \
    fi

# 4) Extra build-time Python deps for extensions (safe to repeat numpy headers)
RUN pip install --no-cache-dir \
    cython==3.0.10 numpy==1.24.4 packaging==23.2 ninja


# Create torch extensions cache dir (writable)
RUN mkdir -p "$TORCH_EXTENSIONS_DIR"

# Copy the rest of your project into the image
COPY . .

# Build your project’s native/CUDA extensions
# FPN
RUN set -e; \
    if [ -f "lib/fpn/make.sh" ]; then \
      echo "Building lib/fpn ..."; \
      cd lib/fpn && bash make.sh && cd -; \
    fi

# Caches (optional but helpful)
ENV HF_HOME=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/.cache/wandb
ARG GROUPID=${USERID}
RUN mkdir -p $HF_HOME $WANDB_DIR && chown -R ${USERID}:${GROUPID} /workspace

USER ${USERNAME}

# Good signal handling
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]
