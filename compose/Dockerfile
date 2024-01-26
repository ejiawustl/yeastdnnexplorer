# Start from an NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 as base

# Install Python 3.11
RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y python3.11 python3.11-distutils python3.11-venv && \
  ln -s /usr/bin/python3.11 /usr/local/bin/python3 && \
  ln -s /usr/bin/python3.11 /usr/local/bin/python && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install system dependencies
# note that these may not be necessary and may already be in the base image
# left here b/c i anticipate adding HTSlib
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  zlib1g-dev \
  libbz2-dev \
  liblzma-dev \
  curl \
  git \
  sudo \
  bash-completion \
  vim \
  ssh \
  && rm -rf /var/lib/apt/lists/*

# Create devcontainer user and add it to sudoers
RUN groupadd --gid 1000 dev-user \
  && useradd --uid 1000 --gid dev-user --shell /bin/bash --create-home dev-user \
  && echo dev-user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/dev-user \
  && chmod 0440 /etc/sudoers.d/dev-user

# Set environment variables to non-interactive (this prevents some prompts)
ENV POETRY_NO_INTERACTION=1

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python -

# Add Poetry to PATH directly
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the Docker image
WORKDIR /app

# Copy only the pyproject.toml and poetry.lock files to install dependencies
# This enables caching of dependencies to speed up builds
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies via Poetry
RUN poetry install

# Copy the rest of your project files into the Docker image
COPY . /app

# Run the container with volume mapping for live code reloading
# docker run -v $(pwd):/app <your_image_name>
# note that this is already configured in the docker compose gpu.yml and
# cpu.yml files
