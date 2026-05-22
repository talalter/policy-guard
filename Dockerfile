FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first — saves ~1.5 GB vs the default CUDA build.
# Remaining packages are installed separately so this layer is cached independently.
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

RUN grep -v "^torch==" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Pre-download NLTK data so the first request doesn't block on a network call.
# punkt_tab is the tokenizer used by split_sentences(); stopwords is used by the lexical gate.
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Pre-bake HuggingFace model weights into the image.
# Keeps startup time fast — models are loaded from the image layer, not downloaded at runtime.
# These two layers are cached as long as the model names in config.py don't change.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('dleemiller/ModernCE-base-nli'); \
AutoModelForSequenceClassification.from_pretrained('dleemiller/ModernCE-base-nli')"

# Copy application code last — changes here only invalidate this final layer.
COPY backend/ backend/

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
