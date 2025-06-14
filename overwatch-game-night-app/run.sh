#!/bin/bash

echo "Starting Overwatch Game Night App..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the application
echo "Starting server on http://localhost:8599"
echo "Share with friends: http://YOUR_IP:8599"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
