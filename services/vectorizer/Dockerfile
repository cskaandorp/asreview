FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first to avoid sentence-transformers pulling the
# full CUDA-enabled wheel (~1.5 GB) we don't need on a CPU-only NUC.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so first container start doesn't pay the
# 600+ MB download cost at runtime. Adds ~700 MB to the image.
ARG HF_MODEL_ID=mixedbread-ai/mxbai-embed-large-v1
ENV HF_MODEL_ID=${HF_MODEL_ID}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${HF_MODEL_ID}')"

COPY worker.py .

CMD ["python", "-u", "worker.py"]
