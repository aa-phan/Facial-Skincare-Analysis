# Use a lightweight Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*


