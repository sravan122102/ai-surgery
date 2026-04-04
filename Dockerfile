# Use the official PyTorch base image representing an Ubuntu environment
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies required for OpenCV and other media handling
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
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
