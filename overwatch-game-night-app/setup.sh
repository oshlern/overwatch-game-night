#!/bin/bash

echo "Setting up Overwatch Game Night App..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Import existing data
echo "Importing existing player data..."
python import_existing_data.py

echo ""
echo "Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py"
echo "3. Open your browser to: http://localhost:8599"
echo ""
echo "To share with friends on the same network:"
echo "Find your IP address and share: http://YOUR_IP:8599"
