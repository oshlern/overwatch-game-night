# Overwatch Game Night App

A Flask-based web application for hosting Overwatch game nights with team balancing, arcade game picking, and award voting.

## Features

- **Team Balancing**: Smart algorithm to create fair 6v6 teams based on SR and role preferences
- **Game Picker**: Random arcade game selection with game history tracking
- **Awards System**: Real-time voting for best plays with token rewards
- **Local Hosting**: Run locally and share with friends on the same network
- **Persistent Data**: All data stored locally in JSON files

## Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
cd overwatch-game-night-app
./setup.sh
```

### Option 2: Manual Setup
1. **Create Virtual Environment**
   ```bash
   cd overwatch-game-night-app
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Import Existing Data (Optional)**
   ```bash
   python import_existing_data.py
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the App**
   - Open your browser to `http://localhost:8599`
   - Share your local IP address with friends (e.g., `http://192.168.1.100:8599`)

## Data Files

The app creates a `data/` directory with these files:
- `players.json` - Player data with SR and role preferences
- `gamemodes.json` - Available arcade games and custom modes
- `games_history.json` - Log of played games
- `matches.json` - Team balancing history
- `tokens.json` - Player token counts
- `votes.json` - Current voting state

## Importing Existing Data

To import your existing player data from CSV:
1. Go to Team Balance page
2. Use the "Import Players" section
3. Upload a CSV with format: `bnet, discord, Tank SR, DPS SR, Support SR, Wants Tank, Wants DPS, Wants Support`

## Award Categories

- **Most Improved on New Hero**: For trying new heroes and doing well
- **Biggest Clutch**: Game-changing plays that turned the tide
- **Most Stylish Play/Skin**: Best plays or coolest skin combos

## Network Setup

For friends to connect:
1. Make sure everyone is on the same WiFi network
2. Find your computer's IP address
3. Share the address: `http://YOUR_IP:8599`
4. Friends can join by visiting that URL

## Troubleshooting

- **Can't connect**: Check firewall settings and ensure port 8599 is open
- **Teams not balanced**: Make sure you have at least 10 players selected
- **Voting not working**: Check that WebSocket connections are working

## Development

Built with:
- Flask (web framework)
- Flask-SocketIO (real-time features)
- Bootstrap 5 (UI)
- CVXPY (optimization for team balancing)
- Pandas/NumPy (data processing)




TODO:
make arcade games tailored to number of players
make admin tab, with ability to close voting. Votes should be hidden until voting is closed. Allow people to type explanation in nomination
Keep tab of active players always on the screen (on the left) with abilitiy to set as afk (like in old code), nominate, etc.

Make more game themed and arcade retro purple theme
