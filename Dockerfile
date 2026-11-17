FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY n4c_decision_broker ./n4c_decision_broker
RUN pip install ".[server]"

# Non-root user; audit log and decision cache live in a volume.
RUN useradd --system --uid 10001 --home-dir /data broker \
    && mkdir -p /data/audit && chown -R broker /data
USER broker
WORKDIR /data
VOLUME /data/audit

# Ollama runs outside the container by default (on the host).
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/v1/broker/health', timeout=2).status==200 else 1)"

# N4C_BROKER_API_KEYS is required: the server refuses to bind 0.0.0.0 without it.
ENTRYPOINT ["n4c-decision-broker"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
