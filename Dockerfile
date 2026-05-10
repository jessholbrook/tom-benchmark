# Lightweight container for the Streamlit dashboard.
# Works on Fly.io, Hugging Face Spaces (Docker SDK), Render, Railway, etc.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt ./
COPY tom_benchmark ./tom_benchmark

RUN pip install --upgrade pip && \
    pip install -e ".[dashboard]"

# Now copy the rest of the app.
COPY app.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

# Streamlit listens on $PORT if provided (Fly, Cloud Run), otherwise 8501.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
