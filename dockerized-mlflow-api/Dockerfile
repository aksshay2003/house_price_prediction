FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies including SSL support
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
