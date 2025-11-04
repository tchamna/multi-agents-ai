# Multi-Agent AI - Dockerfile
# Use slim Python base image
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Copy minimal requirements first so we can install them in a single layer
# along with system build deps and then remove those build deps — this
# prevents large build toolchains from remaining in the final image layers.
COPY requirements_minimal.txt ./

# Install system build deps, install Python packages, then purge build deps
# and remove caches in the same RUN to keep the image small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git wget curl ca-certificates && \
    pip install --no-cache-dir -r requirements_minimal.txt && \
    apt-get purge -y --auto-remove build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip /root/.cache/huggingface

# Copy application
COPY . /app

# Use non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Streamlit default port; allow override via PORT env var
ENV PORT=8501
EXPOSE 8501

# Make logs unbuffered
ENV PYTHONUNBUFFERED=1

# Run Streamlit
# Bind to 0.0.0.0 inside container (for port mapping to work from host)
# Suppress misleading 0.0.0.0 URL output and show startup time in hh:mm:ss format
CMD sh -c "START_TIME=\$(date +%s); echo '======================================'; echo 'Streamlit app is starting...'; echo '======================================'; echo ''; echo 'Open your browser and go to:'; echo '  http://localhost:$PORT'; echo ''; /usr/local/bin/python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --logger.level error --client.showErrorDetails false 2>&1 | grep -v 'URL: http://0.0.0.0' | grep -v 'You can now view' || true & PID=\$!; wait \$PID; END_TIME=\$(date +%s); ELAPSED=\$((END_TIME - START_TIME)); HOURS=\$((ELAPSED / 3600)); MINS=\$(((ELAPSED % 3600) / 60)); SECS=\$((ELAPSED % 60)); printf '\n======================================\n'; printf 'Running time: %02d:%02d:%02d\n' \$HOURS \$MINS \$SECS; printf '======================================\n'"
