#!/bin/bash

# Scientific Paper Summarization API - Stop Server Script
# This script stops the running Gunicorn server

PID_FILE="./logs/gunicorn.pid"

echo "Stopping Scientific Paper Summarization API..."

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping server with PID: $PID"
        kill $PID
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force stopping server..."
            kill -9 $PID
        fi
        
        # Remove PID file
        rm -f "$PID_FILE"
        echo "✅ Scientific Paper Summarization API stopped successfully!"
    else
        echo "⚠️  Process with PID $PID is not running"
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  PID file not found. Trying to find and stop any running processes..."
    
    # Find and kill any running gunicorn processes for this app
    PIDS=$(pgrep -f "gunicorn.*summarizer_api")
    if [ -n "$PIDS" ]; then
        echo "Found running processes: $PIDS"
        echo $PIDS | xargs kill
        sleep 2
        
        # Force kill if still running
        REMAINING=$(pgrep -f "gunicorn.*summarizer_api")
        if [ -n "$REMAINING" ]; then
            echo "Force stopping remaining processes..."
            echo $REMAINING | xargs kill -9
        fi
        echo "✅ Stopped running processes"
    else
        echo "ℹ️  No running Scientific Paper Summarization API processes found"
    fi
fi
