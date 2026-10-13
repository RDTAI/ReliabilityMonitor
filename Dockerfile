FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /app/state
EXPOSE 8000
CMD ["uvicorn", "reliability_monitor.api:app", "--host", "0.0.0.0", "--port", "8000"]

