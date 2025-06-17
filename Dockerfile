# NVIDIA CUDA 베이스 이미지 사용 (CUDA 12.1, cuDNN 8 포함)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 1. 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. python, pip 최신화
RUN ln -s /usr/bin/python3 /usr/bin/python && pip install --upgrade pip

# 3. PyTorch 설치 (CUDA 12.1 빌드)
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 필수 라이브러리 설치
RUN pip install pytorch-lightning transformers tokenizers sentencepiece shortuuid accelerate peft \
    bitsandbytes pydantic numpy==1.26.4 scikit-learn gradio gradio_client spaces \
    requests httpx uvicorn fastapi einops==0.6.1 einops-exts timm tensorflow==2.15.1 \
    tensorflow_datasets tensorflow_graphics draccus pyav numba loguru sacrebleu evaluate \
    sqlitedict open_clip_torch

# 5. GitHub에서 dlimp 설치
RUN pip install git+https://github.com/moojink/dlimp_openvla

# 6. flash-attn 설치 (만약 필요하면)
# (여기서 소스 컴파일 필요 -> CUDA toolkit 설치된 베이스이므로 문제 없음)
RUN pip install flash-attn --no-build-isolation

# 7. 추가: train / eval / agent 옵션 종속성도 같이 설치
RUN pip install deepspeed
RUN pip install ninja wandb
RUN pip install azure-ai-ml datasets fire openai==1.8.0
RUN pip install opencv-python openpyxl==3.1.2 pillow==9.4.0 python-Levenshtein
RUN pip install rich streamlit==1.29.0 typer[all] word2number
RUN pip install pygame easyocr paddleocr
RUN pip install common==0.1.2 dual==0.0.10 tight==0.1.0 prox==0.0.17
RUN pip install data
RUN pip install paddle paddlepaddle
RUN pip install supervision==0.18.0 ultralytics==8.3.78
# 8. 기본 작업 디렉토리 설정
WORKDIR /workspace

# 9. 기본 명령어
CMD ["bash"]

