FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12.16-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /workspace/casepilot
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY . .
RUN python scripts/bootstrap_retail.py && uv sync --frozen --no-dev
EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
