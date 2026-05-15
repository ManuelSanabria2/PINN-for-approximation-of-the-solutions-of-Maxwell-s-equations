# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables
ENV PORT=7860

# Run from the source folder because the project directory contains a hyphen.
WORKDIR /app/pinn-maxwell
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
