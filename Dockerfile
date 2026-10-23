FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# ${PORT:-8000}: PaaS hosts (Render, HF Spaces, Railway...) assign the port via
# $PORT; default keeps `docker compose up` working unchanged locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
