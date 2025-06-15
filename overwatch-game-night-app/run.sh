#!/bin/bash

echo "Starting Overwatch Game Night App..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start ngrok tunnel
echo "Starting ngrok tunnel..."
ngrok http --url=only-prime-gibbon.ngrok-free.app 8599 --config=ngrok.yml &

# Run the application
echo "Starting server on http://localhost:8599"
echo "Share with friends: http://YOUR_IP:8599"
echo "Public URL: https://only-prime-gibbon.ngrok-free.app"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
