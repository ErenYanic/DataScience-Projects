# 1. Base Image: Lightweight Python 3.13
# Using 'slim' variant to keep the image size small while having necessary tools.
# This provides the OS layer (Linux) and Python runtime.
FROM python:3.13-slim

# 2. System Dependencies: Install essential C++ libraries
# libgomp1 is often required for XGBoost/LightGBM/CatBoost parallel processing.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory: Set the internal directory where our app will live
WORKDIR /app

# 4. Dependency Installation
# Copy only requirements first to leverage Docker layer caching.
COPY requirements.txt .

# Install Python packages using pip
# We upgrade pip first to ensure compatibility with 3.13 wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Application Code
# Copy the rest of the source code and model artefacts into the container
COPY . .

# 6. Exposure
# Inform Docker that the container listens on port 8000
EXPOSE 8000

# 7. Entry Point
# The command to execute when the container starts.
# We launch the FastAPI server using the 'python src/main.py' logic.
CMD ["python", "src/main.py"]
