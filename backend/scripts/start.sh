#!/bin/bash

# Aero Melody Backend Startup Script

echo "Starting Aero Melody Backend..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using default configuration."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p midi_output
mkdir -p logs

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Start the application
echo "Starting FastAPI application..."
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info \
    --access-log
