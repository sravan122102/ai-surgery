# Use the official Ultralytics image which already includes PyTorch + ultralytics
# This avoids pip installing ~1GB of dependencies during build
FROM ultralytics/ultralytics:latest-python

# Install system dependencies required for OpenCV and ffmpeg
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Add a user to avoid running as root, required by Hugging Face Spaces
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/app/uploads /home/user/app/results && \
    chown -R user:user /home/user/app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install only the additional (light) dependencies not in the base image
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . .

# Expose the standard Hugging Face Docker Space port
EXPOSE 7860

# Command to run Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "300", "--workers", "1", "--threads", "4"]
