FROM python:3.9-slim

# Install dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY main.py .

# Copy the model from the volume (Mounted outside Docker)
VOLUME /mnt

# Run the application
CMD ["python", "main.py"]
