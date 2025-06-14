#!/bin/bash

# Scientific Paper Summarization API - Production Deployment Script
# This script starts the FastAPI application using Gunicorn with optimized settings

# Configuration
WORKERS=4
WORKER_CLASS="uvicorn.workers.UvicornWorker"
BIND_ADDRESS="0.0.0.0:8000"
APP_MODULE="summarizer_api:app"

# Log files
ACCESS_LOG="./logs/summarizer_api_access.log"
ERROR_LOG="./logs/summarizer_api_error.log"
LOG_LEVEL="info"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the application
gunicorn \
  --workers $WORKERS \
  --worker-class $WORKER_CLASS \
  --bind $BIND_ADDRESS \
  --daemon \
  --access-logfile $ACCESS_LOG \
  --error-logfile $ERROR_LOG \
  --log-level $LOG_LEVEL \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --timeout 300 \
  --keep-alive 2 \
  --pid ./logs/gunicorn.pid \
  $APP_MODULE

echo "Scientific Paper Summarization API started successfully!"
echo "PID file: ./logs/gunicorn.pid"
echo "Access log: $ACCESS_LOG"
echo "Error log: $ERROR_LOG"
echo "API available at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo "Health check available at: http://localhost:8000/health"
echo ""
echo "Management commands:"
echo "  - Check health: ./health_check.sh"
echo "  - Stop server: ./stop_server.sh"