#!/bin/bash
set -e

echo "🌍 Starting Civilization Backend..."

if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "👉  Edit .env and add your CEREBRAS_API_KEY, then run again."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

if [ -z "$CEREBRAS_API_KEY" ] || [ "$CEREBRAS_API_KEY" = "your_cerebras_api_key_here" ]; then
    echo "❌  CEREBRAS_API_KEY not set in .env"
    exit 1
fi

echo "✓ API key loaded"
echo "🚀 Launching on http://localhost:8000"
echo "🔌 WebSocket at ws://localhost:8000/ws"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload