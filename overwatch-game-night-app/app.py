import json
import os
import uuid
from datetime import datetime

import cvxpy as cp
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "overwatch-game-night-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*")

# Data storage paths
DATA_DIR = "data"
PLAYERS_FILE = os.path.join(DATA_DIR, "players.json")
GAMEMODES_FILE = os.path.join(DATA_DIR, "gamemodes.json")
GAMES_HISTORY_FILE = os.path.join(DATA_DIR, "games_history.json")
TOKENS_FILE = os.path.join(DATA_DIR, "tokens.json")
MATCHES_FILE = os.path.join(DATA_DIR, "matches.json")
VOTES_FILE = os.path.join(DATA_DIR, "votes.json")
AWARD_HISTORY_FILE = os.path.join(DATA_DIR, "award_history.json")
CATEGORIES_FILE = os.path.join(DATA_DIR, "categories.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize data files if they don't exist
def init_data_files():
    if not os.path.exists(PLAYERS_FILE):
        with open(PLAYERS_FILE, "w") as f:
            json.dump({}, f)

    if not os.path.exists(GAMEMODES_FILE):
        # Initialize with some popular games
        default_games = [
            {
                "id": "43KZV",
                "code": "43KZV",
                "name": "Route 66",
                "description": "Classic payload map",
                "category": "Arcade",
                "recommended_time": 15,
            },
            {
                "id": "ATAHA",
                "code": "ATAHA",
                "name": "Havana",
                "description": "Escort through the streets",
                "category": "Arcade",
                "recommended_time": 15,
            },
            {
                "id": "MYSTERY",
                "code": "MYSTERY",
                "name": "Mystery Heroes",
                "description": "Random hero each spawn",
                "category": "Arcade",
                "recommended_time": 10,
            },
        ]
        with open(GAMEMODES_FILE, "w") as f:
            json.dump(default_games, f)

    if not os.path.exists(GAMES_HISTORY_FILE):
        with open(GAMES_HISTORY_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "w") as f:
            json.dump({}, f)

    if not os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(VOTES_FILE):
        with open(VOTES_FILE, "w") as f:
            json.dump({}, f)

    if not os.path.exists(AWARD_HISTORY_FILE):
        with open(AWARD_HISTORY_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "w") as f:
            json.dump([], f)


init_data_files()

# Team matching weights (from existing code)
WEIGHTS = {
    "AVG_SR_WEIGHT": 0.15,
    "SR_DIFF_PENALTY_WEIGHT": 0.8,
    "MAX_SR_DIFF": 5,
    "AVG_COMF_WEIGHT": 1.0,
}




def generate_player_color():
    """Generate a vibrant color for a new player"""
    import random

    # Predefined vibrant colors that work well on dark backgrounds
    vibrant_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E9",
        "#F8C471",
        "#82E0AA",
        "#F1948A",
        "#D7BDE2",
        "#A9DFBF",
        "#F9E79F",
        "#D2B4DE",
        "#AED6F1",
        "#FADBD8",
        "#D5F4E6",
        "#FCF3CF",
        "#EBDEF0",
        "#D6EAF8",
        "#D1F2EB",
        "#FEF9E7",
        "#FDEDEC",
        "#EAF2F8",
        "#FF9F43",
        "#10AC84",
        "#EE5A24",
        "#0984E3",
        "#A29BFE",
        "#FD79A8",
        "#FDCB6E",
        "#6C5CE7",
        "#74B9FF",
    ]

    # Get existing colors to avoid duplicates
    players = load_json_file(PLAYERS_FILE)
    used_colors = {
        player.get("color") for player in players.values() if player.get("color")
    }

    # Find available colors
    available_colors = [color for color in vibrant_colors if color not in used_colors]

    # If all predefined colors are used, generate a random one
    if not available_colors:
        # Generate a random vibrant color
        hue = random.randint(0, 360)
        saturation = random.randint(70, 100)
        lightness = random.randint(50, 80)

        # Convert HSL to RGB (simplified)
        import colorsys

        r, g, b = colorsys.hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return random.choice(available_colors)


def load_json_file(filepath):
    """Load JSON data from file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if filepath in [PLAYERS_FILE, TOKENS_FILE, VOTES_FILE] else []


def save_json_file(filepath, data):
    """Save JSON data to file"""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def get_user_id():
    """Get or create user ID for session"""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def load_categories():
    """Load categories from file"""
    return load_json_file(CATEGORIES_FILE)


def save_categories(categories):
    """Save categories to file"""
    save_json_file(CATEGORIES_FILE, categories)


def add_category(category_name, description=""):
    """Add a new category if it doesn't exist"""
    categories = load_categories()

    # Check if category already exists (by name)
    existing_names = [cat.get("name") if isinstance(cat, dict) else cat for cat in categories]
    if category_name not in existing_names:
        new_category = {
            "name": category_name,
            "description": description
        }
        categories.append(new_category)
        save_categories(categories)
    return categories


@app.route("/")
def home():
    """Home page with navigation"""
    return render_template("home.html")


@app.route("/team-balance")
def team_balance():
    """Team balancing page"""
    players = load_json_file(PLAYERS_FILE)
    return render_template("team_balance.html", players=players)


@app.route("/arcade-picker")
def arcade_picker():
    """Arcade game picker page"""
    games = load_json_file(GAMEMODES_FILE)
    return render_template("arcade_picker.html", games=games)


@app.route("/awards")
def awards():
    """Award voting page"""
    players = load_json_file(PLAYERS_FILE)
    tokens = load_json_file(TOKENS_FILE)
    categories = load_categories()
    return render_template(
        "awards.html", players=players, tokens=tokens, categories=categories
    )


@app.route("/admin")
def admin():
    """Admin control panel"""
    players = load_json_file(PLAYERS_FILE)
    tokens = load_json_file(TOKENS_FILE)
    votes = load_json_file(VOTES_FILE)
    categories = load_categories()
    return render_template(
        "admin.html",
        players=players,
        tokens=tokens,
        votes=votes,
        categories=categories,
    )


@app.route("/api/players", methods=["GET", "POST"])
def api_players():
    """API endpoint for player management"""
    if request.method == "GET":
        return jsonify(load_json_file(PLAYERS_FILE))

    elif request.method == "POST":
        players = load_json_file(PLAYERS_FILE)
        player_data = request.json

        # Add timestamp
        player_data["last_updated"] = datetime.now().isoformat()

        # Assign color if this is a new player or if color is missing
        if player_data["bnet"] not in players or "color" not in players.get(
            player_data["bnet"], {}
        ):
            player_data["color"] = generate_player_color()

        # Store player data
        players[player_data["bnet"]] = player_data
        save_json_file(PLAYERS_FILE, players)

        return jsonify({"status": "success", "message": "Player data saved"})


@app.route("/api/balance-teams", methods=["POST"])
def api_balance_teams():
    """API endpoint for team balancing"""
    try:
        data = request.json
        selected_players = data.get("players", [])
        team_format = data.get("format", "5v5")  # Default to 5v5

        min_players = 10 if team_format == "5v5" else 12
        if len(selected_players) < min_players:
            return jsonify(
                {
                    "error": f"Need at least {min_players} players for {team_format} team balancing"
                }
            ), 400

        # Convert to DataFrame format expected by team matching algorithm
        df_data = []
        for player_name in selected_players:
            players = load_json_file(PLAYERS_FILE)
            if player_name in players:
                player = players[player_name]
                df_data.append(
                    {
                        "bnet": player["bnet"],
                        "discord": player.get("discord", player["bnet"]),
                        "Tank SR": player.get("tank_sr", 0),
                        "DPS SR": player.get("dps_sr", 0),
                        "Support SR": player.get("support_sr", 0),
                        "Wants Tank": player.get("wants_tank", 0),
                        "Wants Dps": player.get("wants_dps", 0),
                        "Wants Support": player.get("wants_support", 0),
                    }
                )

        df = pd.DataFrame(df_data)

        # Run team matching algorithm with format
        result_df, avg_srs, devs, comfs = optimize_team_matching(df, team_format)

        # Save match result
        match_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": team_format,
            "players": selected_players,
            "teams": result_df.to_dict("records"),
            "avg_srs": avg_srs,
            "avg_comfort": float(np.mean(comfs)),
        }

        matches = load_json_file(MATCHES_FILE)
        matches.append(match_data)
        save_json_file(MATCHES_FILE, matches)

        return jsonify(
            {
                "teams": result_df.to_dict("records"),
                "metrics": {
                    "avg_comfort": float(np.mean(comfs)),
                    "sr_difference": float(
                        abs(avg_srs[0] - avg_srs[1]) if len(avg_srs) >= 2 else 0
                    ),
                    "overall_avg_sr": float(np.mean(avg_srs)),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def optimize_team_matching(df, team_format="5v5"):
    """Optimized team matching algorithm (adapted from existing code)"""
    # Extract role SRs and comfort preferences
    role_srs = df[["Tank SR", "DPS SR", "Support SR"]].to_numpy()
    role_comfs = df[["Wants Tank", "Wants Dps", "Wants Support"]].to_numpy()

    # Define constants based on team format
    n_teams = 2
    n_roles = 3
    if team_format == "6v6":
        team_size = 6
        tanks_per_team = 2
        dps_per_team = 2
        supports_per_team = 2
    else:  # 5v5
        team_size = 5
        tanks_per_team = 1
        dps_per_team = 2
        supports_per_team = 2

    n_players = len(df)
    id_n = n_teams * n_roles

    # Define optimization variables
    identity = cp.Variable((n_players, id_n), boolean=True)

    # Define constraints
    constraints = [
        cp.sum(identity, axis=1) <= 1
    ]  # A player can be assigned to at most one role

    # Ensure each team has the correct number of roles
    team_roles = cp.sum(identity, axis=0)
    for i in range(n_teams):
        constraints += [team_roles[3 * i] == tanks_per_team]  # Tanks
        constraints += [team_roles[3 * i + 1] == dps_per_team]  # DPS
        constraints += [team_roles[3 * i + 2] == supports_per_team]  # Support

    # Calculate role assignments and comfort scores
    is_role = cp.vstack(
        [
            sum(identity[:, i + j] for i in range(0, id_n, n_roles))
            for j in range(n_roles)
        ]
    ).T
    comfs = cp.sum(cp.multiply(role_comfs, is_role), axis=1)
    constraints += [comfs >= 0]

    # Calculate team SRs and ensure SR deviations are within the allowed range
    team_total_srs = cp.hstack(
        [
            cp.sum(cp.multiply(identity[:, i : i + n_roles], role_srs))
            for i in range(0, id_n, n_roles)
        ]
    )
    team_avg_srs = team_total_srs / team_size
    avg_sr = cp.sum(team_avg_srs) / n_teams
    sr_deviations = team_avg_srs - avg_sr
    constraints += [
        sr_deviations <= WEIGHTS["MAX_SR_DIFF"],
        sr_deviations >= -WEIGHTS["MAX_SR_DIFF"],
    ]

    # Define the objective function to maximize average comfort and balance SR
    avg_comf = cp.sum(comfs) / (n_teams * team_size)
    cost = WEIGHTS["AVG_COMF_WEIGHT"] * avg_comf
    cost += WEIGHTS["AVG_SR_WEIGHT"] * avg_sr
    cost -= WEIGHTS["SR_DIFF_PENALTY_WEIGHT"] * (
        cp.max(team_avg_srs) - cp.min(team_avg_srs)
    )
    objective = cp.Maximize(cost)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the results
    ids = identity.value
    is_role = np.array(
        [sum(ids[:, i + j] for i in range(0, id_n, n_roles)) for j in range(n_roles)]
    ).T
    is_team = np.array(
        [sum(ids[:, i + j] for j in range(n_roles)) for i in range(0, id_n, n_roles)]
    ).T
    is_role = np.rint(is_role).astype(int)
    is_team = np.rint(is_team).astype(int)

    # Determine team and role assignments
    teams = np.where(np.all(is_team == 0, axis=1), -1, np.argmax(is_team, axis=1))
    roles = np.where(np.all(is_role == 0, axis=1), -1, np.argmax(is_role, axis=1))

    # Calculate SRs and comfort scores for each player
    srs = np.squeeze(np.take_along_axis(role_srs, roles[:, None], axis=1))
    comfs = np.squeeze(np.take_along_axis(role_comfs, roles[:, None], axis=1))
    avg_srs = [np.mean(srs[teams == i]) for i in range(n_teams)]

    # Create a DataFrame with the matchings
    result_df = pd.DataFrame(
        {
            "Battletag": df["bnet"],
            "Discord": df["discord"],
            "SR": [srs[i] for i in range(len(srs))],
            "Comf": [comfs[i] for i in range(len(comfs))],
            "Role": [
                ["Tank", "DPS", "Support"][roles[i]] if roles[i] != -1 else "Bench"
                for i in range(len(roles))
            ],
            "Team": [teams[i] for i in range(len(teams))],
        }
    ).sort_values(["Team", "Role"])

    # Remove rows where teams is -1
    result_df = result_df[result_df["Team"] != -1]

    return result_df, avg_srs, None, comfs


@app.route("/api/random-arcade")
def api_random_arcade():
    """API endpoint for random arcade game selection"""
    num_players = request.args.get("players", type=int)
    games = load_json_file(GAMEMODES_FILE)

    if games:
        # Filter games by player count if specified
        if num_players:
            suitable_games = []
            for game in games:
                # Parse player requirements from description or use defaults
                if "any" in str(game.get("description", "")).lower():
                    suitable_games.append(game)
                elif "at least" in str(game.get("description", "")).lower():
                    # Extract minimum player requirement
                    import re

                    match = re.search(
                        r"at least (\d+)", str(game.get("description", "")).lower()
                    )
                    if match and num_players >= int(match.group(1)):
                        suitable_games.append(game)
                else:
                    # Default: include all games
                    suitable_games.append(game)

            if suitable_games:
                selected_game = suitable_games[
                    np.random.randint(0, len(suitable_games))
                ]
            else:
                selected_game = games[np.random.randint(0, len(games))]
        else:
            selected_game = games[np.random.randint(0, len(games))]

        return jsonify(selected_game)
    return jsonify({"error": "No games available"}), 404


@app.route("/api/gamemodes")
def api_gamemodes():
    """API endpoint for getting all gamemodes"""
    games = load_json_file(GAMEMODES_FILE)
    return jsonify(games)


@app.route("/api/games-history", methods=["GET", "POST"])
def api_games_history():
    """API endpoint for getting and posting games history"""
    if request.method == "GET":
        history = load_json_file(GAMES_HISTORY_FILE)
        return jsonify(history)

    elif request.method == "POST":
        # Add new game to history
        game_data = request.json
        history = load_json_file(GAMES_HISTORY_FILE)

        # Add timestamp if not present
        if "timestamp" not in game_data:
            game_data["timestamp"] = datetime.now().isoformat()

        # Add unique ID if not present
        if "id" not in game_data:
            game_data["id"] = str(uuid.uuid4())

        history.append(game_data)
        save_json_file(GAMES_HISTORY_FILE, history)

        return jsonify({"status": "success", "message": "Game logged successfully"})


@app.route("/api/players/status", methods=["POST"])
def api_player_status():
    """Update player status (active/afk)"""
    data = request.json
    player_name = data.get("player")
    status = data.get("status")  # 'active' or 'afk'

    players = load_json_file(PLAYERS_FILE)
    if player_name in players:
        players[player_name]["status"] = status
        players[player_name]["last_updated"] = datetime.now().isoformat()
        save_json_file(PLAYERS_FILE, players)

        # Broadcast status update
        socketio.emit(
            "player_status_update",
            {"player": player_name, "status": status},
            room="voting",
        )

        return jsonify({"status": "success"})

    return jsonify({"error": "Player not found"}), 404


@app.route("/api/admin/close-voting", methods=["POST"])
def api_close_voting():
    """Close voting for a category and reveal results"""
    data = request.json
    category = data.get("category")

    votes = load_json_file(VOTES_FILE)

    if category in votes and votes[category].get("votes"):
        # Calculate winner
        vote_counts = {
            player: len(voters) for player, voters in votes[category]["votes"].items()
        }
        winner = max(vote_counts, key=vote_counts.get)
        max_votes = vote_counts[winner]

        # Award token
        tokens = load_json_file(TOKENS_FILE)
        if winner not in tokens:
            tokens[winner] = 0
        tokens[winner] += 1
        save_json_file(TOKENS_FILE, tokens)

        # Save to award history
        award_history = load_json_file(AWARD_HISTORY_FILE)
        award_entry = {
            "winner": winner,
            "category": category,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "votes": max_votes,
            "timestamp": datetime.now().isoformat(),
        }
        award_history.append(award_entry)
        save_json_file(AWARD_HISTORY_FILE, award_history)

        # Broadcast results
        socketio.emit(
            "voting_closed",
            {
                "category": category,
                "winner": winner,
                "votes": max_votes,
                "all_votes": vote_counts,
            },
            room="voting",
        )

        # Reset votes for this category
        votes[category] = {"nominations": {}, "votes": {}}
        save_json_file(VOTES_FILE, votes)

        return jsonify({"status": "success", "winner": winner, "votes": max_votes})

    return jsonify({"error": "No votes to close"}), 400


@app.route("/api/voting-state")
def api_voting_state():
    """Get current voting state (nominations and votes)"""
    votes = load_json_file(VOTES_FILE)
    return jsonify(votes)


@app.route("/api/admin/reset-category", methods=["POST"])
def api_reset_category():
    """Reset nominations and votes for a specific category"""
    data = request.json
    category = data.get("category")

    if not category:
        return jsonify({"error": "Category required"}), 400

    votes = load_json_file(VOTES_FILE)

    # Reset the category
    votes[category] = {"nominations": {}, "votes": {}, "explanations": {}}
    save_json_file(VOTES_FILE, votes)

    # Broadcast reset to all clients
    socketio.emit("category_reset", {"category": category}, room="voting")

    return jsonify(
        {"status": "success", "message": f"Category '{category}' has been reset"}
    )


@app.route("/api/tokens")
def api_tokens():
    """Get current token counts for all players"""
    tokens = load_json_file(TOKENS_FILE)
    return jsonify(tokens)


@app.route("/api/award-history")
def api_award_history():
    """Get award history for slideshow"""
    award_history = load_json_file(AWARD_HISTORY_FILE)

    # If no real history exists, return empty array
    if not award_history:
        return jsonify([])

    # Sort by timestamp (most recent first) and return last 10
    award_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(award_history[:10])


@app.route("/api/admin/adjust-tokens", methods=["POST"])
def api_adjust_tokens():
    """Adjust tokens for a player"""
    data = request.json
    player_name = data.get("player")
    change = data.get("change", 0)

    if not player_name:
        return jsonify({"error": "Player name required"}), 400

    try:
        change = int(change)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid change value"}), 400

    tokens = load_json_file(TOKENS_FILE)

    # Initialize player tokens if they don't exist
    if player_name not in tokens:
        tokens[player_name] = 0

    # Apply change
    new_total = tokens[player_name] + change

    # Don't allow negative tokens
    if new_total < 0:
        return jsonify({"error": "Cannot have negative tokens"}), 400

    tokens[player_name] = new_total
    save_json_file(TOKENS_FILE, tokens)

    return jsonify(
        {
            "status": "success",
            "player": player_name,
            "change": change,
            "new_total": new_total,
        }
    )


@app.route("/api/categories", methods=["GET", "POST"])
def api_categories():
    """API endpoint for category management"""
    if request.method == "GET":
        return jsonify(load_categories())

    elif request.method == "POST":
        data = request.json
        category_name = data.get("name", "").strip()
        category_description = data.get("description", "").strip()

        if not category_name:
            return jsonify({"error": "Category name required"}), 400

        categories = add_category(category_name, category_description)
        return jsonify({"status": "success", "categories": categories})


# WebSocket events for real-time voting
@socketio.on("join_voting")
def on_join_voting(data=None):
    """Handle user joining voting room"""
    user_id = get_user_id()
    join_room("voting")
    emit("user_joined", {"user_id": user_id}, room="voting")


@socketio.on("nominate_player")
def on_nominate_player(data):
    """Handle player nomination with explanation"""
    user_id = get_user_id()
    category = data["category"]
    nominated_player = data["player"]
    explanation = data.get("explanation", "")

    votes = load_json_file(VOTES_FILE)

    # Initialize category if it doesn't exist
    if category not in votes:
        votes[category] = {"nominations": {}, "votes": {}, "explanations": {}}

    # Ensure explanations key exists for existing categories
    if "explanations" not in votes[category]:
        votes[category]["explanations"] = {}

    # Add nomination
    if nominated_player not in votes[category]["nominations"]:
        votes[category]["nominations"][nominated_player] = []

    if nominated_player not in votes[category]["explanations"]:
        votes[category]["explanations"][nominated_player] = []

    # Add nomination if not already nominated by this user
    nomination_entry = {"user_id": user_id, "explanation": explanation}
    existing = [
        n
        for n in votes[category]["explanations"][nominated_player]
        if n["user_id"] == user_id
    ]

    if not existing:
        votes[category]["nominations"][nominated_player].append(user_id)
        votes[category]["explanations"][nominated_player].append(nomination_entry)

    save_json_file(VOTES_FILE, votes)

    # Broadcast nomination update
    emit(
        "nomination_update",
        {
            "category": category,
            "nominations": votes[category]["nominations"],
            "explanations": votes[category]["explanations"],
        },
        room="voting",
    )


@socketio.on("vote_player")
def on_vote_player(data):
    """Handle voting for nominated player"""
    user_id = get_user_id()
    category = data["category"]
    voted_player = data["player"]

    votes = load_json_file(VOTES_FILE)

    if category in votes:
        # Remove previous vote from this user in this category
        for player in votes[category]["votes"]:
            if user_id in votes[category]["votes"][player]:
                votes[category]["votes"][player].remove(user_id)

        # Only add new vote if there's a player to vote for
        if voted_player:
            if voted_player not in votes[category]["votes"]:
                votes[category]["votes"][voted_player] = []
            votes[category]["votes"][voted_player].append(user_id)

        save_json_file(VOTES_FILE, votes)

        # Broadcast vote update (no automatic winner detection)
        emit(
            "vote_update",
            {"category": category, "votes": votes[category]["votes"]},
            room="voting",
        )


@socketio.on("create_category")
def on_create_category(data):
    """Handle creating a new category"""
    category_name = data.get("name", "").strip()
    category_description = data.get("description", "").strip()

    if not category_name:
        return

    # Add category to the categories list
    add_category(category_name, category_description)

    # Broadcast category creation
    emit(
        "category_created",
        {
            "category": category_name,
            "categories": load_categories()
        },
        room="voting",
    )




# WebSocket events for arcade picker synchronization
@socketio.on("join_arcade_picker")
def on_join_arcade_picker():
    """Handle user joining arcade picker room"""
    user_id = get_user_id()
    join_room("arcade_picker")
    emit("user_joined_arcade", {"user_id": user_id}, room="arcade_picker")


@socketio.on("wheel_state_change")
def on_wheel_state_change(data):
    """Handle lobby leader wheel expand/collapse state changes"""
    user_id = get_user_id()
    expanded = data.get("expanded", False)

    # Broadcast to all other users in the arcade picker room
    emit("wheel_state_update", {"expanded": expanded}, room="arcade_picker", include_self=False)


@socketio.on("wheel_spin")
def on_wheel_spin(data):
    """Handle lobby leader wheel spin synchronization"""
    user_id = get_user_id()
    action = data.get("action")

    if action == "start":
        # Broadcast spin start with animation parameters to all other users
        emit("wheel_spin_sync", {
            "action": "start",
            "duration": data.get("duration"),
            "startTime": data.get("startTime"),
            "initialSpeed": data.get("initialSpeed")
        }, room="arcade_picker", include_self=False)
    elif action == "result":
        winner = data.get("winner")
        # Broadcast spin result to all other users
        emit("wheel_spin_sync", {"action": "result", "winner": winner}, room="arcade_picker", include_self=False)


@socketio.on("leader_selected")
def on_leader_selected(data):
    """Handle lobby leader selection synchronization"""
    user_id = get_user_id()
    leader = data.get("leader")

    # Broadcast leader selection to all other users
    emit("leader_selected_sync", {"leader": leader}, room="arcade_picker", include_self=False)


@socketio.on("game_selected")
def on_game_selected(data):
    """Handle game selection synchronization"""
    user_id = get_user_id()
    game_name = data.get("gameName")
    game_data = data.get("gameData")

    # Broadcast game selection to all other users
    emit("game_selected_sync", {"gameName": game_name, "gameData": game_data}, room="arcade_picker", include_self=False)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8599, debug=True)
