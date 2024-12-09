FROM nvidia/cuda:12.2.0-base-ubuntu20.04

WORKDIR /app

# Install necessary system dependencies and CUDA toolkit
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install the CUDA keyring
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA container toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Copy requirements and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model script
COPY model.py .

# Set any necessary environment variables
ENV HUGGINGFACE_HUB_TOKEN=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

# Set the command to run your model
CMD ["python3", "model.py"]
