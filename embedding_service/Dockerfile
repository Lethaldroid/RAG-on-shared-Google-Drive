FROM python:3.11.5-bookworm

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsqlite3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 'credentials.json' file (make sure you have this file)
COPY credentials.json .

COPY token.pickle .

COPY embed.py .

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=credentials.json

CMD ["python", "embed.py"]