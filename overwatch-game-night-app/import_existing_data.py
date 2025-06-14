#!/usr/bin/env python3
"""
Script to import existing player data from the matchings folder
"""

import pandas as pd
import json
import os
import sys

def import_player_data():
    """Import player data from the most recent CSV file"""

    # Look for the most recent player data file
    matchings_dir = '../matchings'
    csv_files = [f for f in os.listdir(matchings_dir) if f.startswith('5v5 - RanksData') and f.endswith('.csv')]

    if not csv_files:
        print("No player data files found in matchings directory")
        return

    # Use the most recent file (highest number)
    latest_file = sorted(csv_files)[-1]
    csv_path = os.path.join(matchings_dir, latest_file)

    print(f"Importing from: {latest_file}")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create players dictionary
        players = {}

        for _, row in df.iterrows():
            if pd.isna(row['bnet']) or row['bnet'] == '':
                continue

            player_data = {
                'bnet': str(row['bnet']),
                'discord': str(row.get('discord', row['bnet'])),
                'tank_sr': float(row.get('Tank SR', 0)) if pd.notna(row.get('Tank SR')) else 0,
                'dps_sr': float(row.get('DPS SR', 0)) if pd.notna(row.get('DPS SR')) else 0,
                'support_sr': float(row.get('Support SR', 0)) if pd.notna(row.get('Support SR')) else 0,
                'wants_tank': int(row.get('Wants Tank', 0)) if pd.notna(row.get('Wants Tank')) else 0,
                'wants_dps': int(row.get('Wants Dps', 0)) if pd.notna(row.get('Wants Dps')) else 0,
                'wants_support': int(row.get('Wants Support', 0)) if pd.notna(row.get('Wants Support')) else 0,
                'imported_from': latest_file
            }

            players[player_data['bnet']] = player_data

        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

        # Save to players.json
        with open('data/players.json', 'w') as f:
            json.dump(players, f, indent=2)

        print(f"Successfully imported {len(players)} players!")
        print("Players imported:")
        for name in players.keys():
            print(f"  - {name}")

    except Exception as e:
        print(f"Error importing data: {e}")

def import_gamemode_data():
    """Import gamemode data from existing CSV if available"""

    gamemodes_csv = '../gamemodes.csv'
    if not os.path.exists(gamemodes_csv):
        print("No gamemodes.csv found, using defaults")
        return

    try:
        df = pd.read_csv(gamemodes_csv)
        games = []

        for _, row in df.iterrows():
            # Skip header row and empty rows
            if pd.isna(row.iloc[1]) or row.iloc[1] == 'Code':
                continue

            code = str(row.iloc[1]).strip()
            name = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else code
            description = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else ''

            # Parse recommended time
            time_str = str(row.iloc[5]) if pd.notna(row.iloc[5]) else '10 mins'
            try:
                # Extract number from time string
                import re
                time_match = re.search(r'(\d+)', time_str)
                recommended_time = int(time_match.group(1)) if time_match else 10
            except:
                recommended_time = 10

            # Determine category
            category = 'Workshop'
            if 'arcade' in name.lower() or code == 'Arcade':
                category = 'Arcade'
            elif 'parkour' in name.lower():
                category = 'Parkour'
            elif any(word in name.lower() for word in ['tag', 'sumo', 'paintball', 'survival']):
                category = 'PvP'
            elif any(word in name.lower() for word in ['mystery', 'elimination', 'deathmatch']):
                category = 'Arcade'

            game_data = {
                'id': code,
                'code': code,
                'name': name,
                'description': description[:100] + '...' if len(description) > 100 else description,
                'category': category,
                'recommended_time': recommended_time
            }
            games.append(game_data)

        # Save to gamemodes.json
        with open('data/gamemodes.json', 'w') as f:
            json.dump(games, f, indent=2)

        print(f"Successfully imported {len(games)} game modes!")
        print("Categories imported:")
        categories = {}
        for game in games:
            cat = game['category']
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in categories.items():
            print(f"  - {cat}: {count} games")

    except Exception as e:
        print(f"Error importing gamemode data: {e}")

if __name__ == '__main__':
    print("Importing existing data...")
    print("=" * 40)

    import_player_data()
    print()
    import_gamemode_data()

    print()
    print("Import complete! You can now run the app with: python app.py")
