FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CFPB_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
# Add `[st]` to the extras to bake in sentence-transformers (much larger image).
RUN pip install --no-cache-dir -e .

# The database and index are NOT baked into the image. Build them once and mount them:
#   docker run -v $(pwd)/data:/data -p 8080:8080 -e ANTHROPIC_API_KEY cfpb-rag
EXPOSE 8080
CMD ["sh", "-c", "uvicorn cfpb_rag.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
