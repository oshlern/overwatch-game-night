import json

# Load the current gamemodes
with open('data/gamemodes.json', 'r') as f:
    games = json.load(f)

# Add player counts to each game based on game type
for game in games:
    name = game['name'].lower()
    description = game['description'].lower()
    category = game['category'].lower()

    # Set player counts based on game characteristics
    if 'keep talking' in name:
        player_count = {'min': 2, 'max': 4, 'optimal': 3}
    elif '1v1' in name or '1v1' in description:
        player_count = {'min': 2, 'max': 2, 'optimal': 2}
    elif '3v3' in name or '3v3' in description:
        player_count = {'min': 6, 'max': 6, 'optimal': 6}
    elif '6v6' in name or '6v6' in description:
        player_count = {'min': 12, 'max': 12, 'optimal': 12}
    elif 'ffa' in name or 'ffa' in description or 'free for all' in description:
        player_count = {'min': 3, 'max': 12, 'optimal': 8}
    elif 'paintball' in name:
        player_count = {'min': 4, 'max': 12, 'optimal': 8}
    elif 'sumo' in name:
        player_count = {'min': 2, 'max': 8, 'optimal': 4}
    elif 'tag' in name:
        player_count = {'min': 3, 'max': 12, 'optimal': 6}
    elif 'hide and seek' in name:
        player_count = {'min': 4, 'max': 12, 'optimal': 8}
    elif 'murder mystery' in name:
        player_count = {'min': 6, 'max': 12, 'optimal': 10}
    elif 'battle royal' in name:
        player_count = {'min': 6, 'max': 12, 'optimal': 10}
    elif 'zombies' in name or 'zombie' in name:
        player_count = {'min': 4, 'max': 12, 'optimal': 6}
    elif 'volleyball' in name or 'lucioball' in name:
        player_count = {'min': 4, 'max': 8, 'optimal': 6}
    elif 'golf' in name or 'pong' in name:
        player_count = {'min': 2, 'max': 8, 'optimal': 4}
    elif 'uno' in name or 'connect four' in name:
        player_count = {'min': 2, 'max': 4, 'optimal': 3}
    elif 'pictionary' in name:
        player_count = {'min': 4, 'max': 12, 'optimal': 8}
    elif 'four square' in name:
        player_count = {'min': 4, 'max': 4, 'optimal': 4}
    elif 'elimination' in name:
        player_count = {'min': 6, 'max': 12, 'optimal': 8}
    elif 'capture the flag' in name:
        player_count = {'min': 6, 'max': 12, 'optimal': 8}
    elif 'mystery heroes' in name:
        player_count = {'min': 6, 'max': 12, 'optimal': 12}
    elif 'deathmatch' in name:
        player_count = {'min': 4, 'max': 12, 'optimal': 8}
    elif category == 'parkour':
        player_count = {'min': 1, 'max': 12, 'optimal': 3}
    elif category == 'arcade':
        player_count = {'min': 6, 'max': 12, 'optimal': 12}
    elif category == 'pvp':
        player_count = {'min': 4, 'max': 12, 'optimal': 8}
    else:
        # Default for workshop games
        player_count = {'min': 2, 'max': 12, 'optimal': 6}

    game['player_count'] = player_count

# Save the updated data
with open('data/gamemodes.json', 'w') as f:
    json.dump(games, f, indent=2)

print(f"Updated {len(games)} games with player count information")
