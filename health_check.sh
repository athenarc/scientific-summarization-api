#!/bin/bash

# Scientific Paper Summarization API - Health Check Script
# This script checks if the API is running and responding properly


if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

HOST=${HOST:-localhost}
PORT=${PORT:-8000}

HEALTH_URL="http://$HOST:$PORT/health"
TIMEOUT=10

echo "Checking API health..."

response=$(curl -s -w "%{http_code}" -o /dev/null --connect-timeout $TIMEOUT "$HEALTH_URL")

if [ "$response" = "200" ]; then
    echo "✅ API is healthy!"
    exit 0
else
    echo "❌ API health check failed (HTTP $response)"
    exit 1
fi
