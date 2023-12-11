FROM python:3.10-slim-bullseye

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache apt-get update && \
    apt-get install -y -qq \
    ffmpeg \
    espeak \
    libportaudio2 \
    build-essential \
    cmake \
    python3-dev \
    portaudio19-dev \
    python3-pyaudio

COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
COPY . .

VOLUME ["/app/models", "/app/songs", "/app/.cache" ]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# webui
EXPOSE 8501

# llm
EXPOSE 8000

# comfyui
EXPOSE 8188

# rvc
EXPOSE 5000

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]