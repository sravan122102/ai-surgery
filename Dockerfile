# Use a lightweight Python base image (much faster build, compatible with HF free tier)
FROM python:3.10-slim

# Install system dependencies required for OpenCV, ffmpeg, and other media handling
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Add a user to avoid running as root, required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . .

# Ensure storage directories exist with proper rights
RUN mkdir -p uploads results

# Expose the standard Hugging Face Docker Space port
EXPOSE 7860

# Command to run Gunicorn referencing the 'app' flask instance in 'app.py'
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "300", "--workers", "1", "--threads", "4"]
