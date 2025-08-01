#!/bin/bash

# Scientific Paper Summarization API - Health Check Script
# This script checks if the API is running and responding properly

API_URL="http://localhost:8000"
HEALTH_URL="http://localhost:8000/health"
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
