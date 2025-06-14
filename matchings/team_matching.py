import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define global variables for cost weights
WEIGHTS = {
    "AVG_SR_WEIGHT": 0.15,
    "SR_DIFF_PENALTY_WEIGHT": 0.8,
    "MAX_SR_DIFF": 5,
    "AVG_COMF_WEIGHT": 1.0  # Add weight for average comfort
}

def read_and_preprocess_data(data_path):
    # Read the data file
    df = pd.read_csv(data_path)

    # Filter out rows where 'Playing' is not NaN and drop any remaining NaN values
    df = df[df['Playing'].notna()]
    df.dropna(inplace=True)
    print(df)

    # Extract role SRs and comfort preferences
    role_srs = df[['Tank SR', 'DPS SR', 'Support SR']].to_numpy()
    role_comfs = df[['Wants Tank', 'Wants Dps', 'Wants Support']].to_numpy()

    # Rename columns for clarity
    df.rename(columns={'bnet': 'bnet', 'discord': 'discord'}, inplace=True)
    ids_df = df[['bnet', 'discord']]

    return df, role_srs, role_comfs, ids_df

def optimize_team_matching(role_srs, role_comfs, ids_df):
    # Define constants
    n_teams = 2
    n_roles = 3
    team_size = 5
    n_players = len(ids_df)
    id_n = n_teams * n_roles

    # Define optimization variables
    identity = cp.Variable((n_players, id_n), boolean=True)

    # Define constraints
    constraints = [cp.sum(identity, axis=1) <= 1]  # A player can be assigned to at most one role

    # Ensure each team has the correct number of roles
    team_roles = cp.sum(identity, axis=0)
    for i in range(n_teams):
        constraints += [team_roles[3*i] == 1]  # 1 Tank
        constraints += [team_roles[3*i+1] == 2]  # 2 DPS
        constraints += [team_roles[3*i+2] == 2]  # 2 Support

    # Calculate role assignments and comfort scores
    is_role = cp.vstack([sum(identity[:, i+j] for i in range(0, id_n, n_roles)) for j in range(n_roles)]).T
    comfs = cp.sum(cp.multiply(role_comfs, is_role), axis=1)
    constraints += [comfs >= 0]

    # Calculate team SRs and ensure SR deviations are within the allowed range
    team_total_srs = cp.hstack([cp.sum(cp.multiply(identity[:, i:i+n_roles], role_srs)) for i in range(0, id_n, n_roles)])
    team_avg_srs = team_total_srs / team_size
    avg_sr = cp.sum(team_avg_srs) / n_teams
    sr_deviations = team_avg_srs - avg_sr
    constraints += [sr_deviations <= WEIGHTS["MAX_SR_DIFF"], sr_deviations >= -WEIGHTS["MAX_SR_DIFF"]]

    # Define the objective function to maximize average comfort and balance SR
    avg_comf = cp.sum(comfs) / (n_teams * team_size)  # Maximize average comfortability
    cost = WEIGHTS["AVG_COMF_WEIGHT"] * avg_comf  # Use weighted average comfort instead of total comfort
    cost += WEIGHTS["AVG_SR_WEIGHT"] * avg_sr  # Add average SR weight
    cost -= WEIGHTS["SR_DIFF_PENALTY_WEIGHT"] * (cp.max(team_avg_srs) - cp.min(team_avg_srs))  # Penalize large SR differences
    objective = cp.Maximize(cost)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the results
    ids = identity.value
    is_role = np.array([sum(ids[:, i+j] for i in range(0, id_n, n_roles)) for j in range(n_roles)]).T
    is_team = np.array([sum(ids[:, i+j] for j in range(n_roles)) for i in range(0, id_n, n_roles)]).T
    is_role = np.rint(is_role).astype(int)
    is_team = np.rint(is_team).astype(int)

    # Determine team and role assignments
    teams = np.where(np.all(is_team == 0, axis=1), -1, np.argmax(is_team, axis=1))
    roles = np.where(np.all(is_role == 0, axis=1), -1, np.argmax(is_role, axis=1))

    # Calculate SRs and comfort scores for each player
    srs = np.squeeze(np.take_along_axis(role_srs, roles[:, None], axis=1))
    comfs = np.squeeze(np.take_along_axis(role_comfs, roles[:, None], axis=1))
    avg_srs = [np.mean(srs[teams == i]) for i in range(n_teams)]
    devs = np.abs(avg_srs - np.mean(srs))

    # Print summary statistics
    print(f"comf: {np.mean(comfs)}, sr: {np.mean(avg_srs)} +- {np.mean(devs)}")
    print(np.max(avg_srs), np.min(avg_srs))
    print(np.std(avg_srs))

    # Create a DataFrame with the matchings
    df = pd.DataFrame({
        "Battletag": ids_df["bnet"],
        "Discord": ids_df["discord"],
        "SR": role_srs.max(axis=1),
        "Comf": role_comfs.max(axis=1),
        "Role": [roles[i] for i in range(len(roles))],
        "Team": [teams[i] for i in range(len(teams))]
    }).sort_values(["Team", "Role"])

    # Remove rows where teams is -1
    df = df[df["Team"] != -1]

    return df, avg_srs, devs, comfs, identity, id_n, n_roles, n_players

def get_team_txt(result_df):
    # Define the role order
    role_order = [0, 1, 1, 2, 2]  # Tank, DPS, DPS, Support, Support

    # Group by teams and create a list of battletags for each team in the specified role order
    grouped_teams = result_df.groupby("Team").apply(
        lambda x: '\t'.join(x.sort_values(by="Role", key=lambda col: col.map({role: i for i, role in enumerate(role_order)}))["Battletag"])
    ).reset_index()

    # Display the grouped teams in a textbox
    grouped_teams_text = '\n'.join(grouped_teams[0])
    return grouped_teams_text

def save_results(df):
    # Save the DataFrame to a CSV file
    df.to_csv("matchings.csv", index=False)

    # Group by teams and create a list of battletags for each team
    grouped_teams = df.groupby("Team")["Battletag"].apply(lambda x: '\t'.join(x)).reset_index()

    # Save only the battletags to a txt file with a row per team, tab separated values, no headers or indices
    grouped_teams["Battletag"].to_csv("grouped_teams.txt", index=False, header=False, sep='\n')

def compute_team_metrics(results_df):
    # Compute the average comfort score
    avg_comf = results_df["Comf"].mean()

    # Compute the average SR for each team
    avg_srs = results_df.groupby("Team")["SR"].mean()

    # Compute the SR difference between the two teams
    sr_diff = abs(avg_srs.max() - avg_srs.min())

    # Compute the overall average SR
    overall_avg_sr = results_df["SR"].mean()

    return avg_comf, sr_diff, overall_avg_sr


def plot_summary_data(n_players, identity, role_comfs, role_srs, avg_sr_weight, sr_diff_penalty_weight, id_n, n_roles, team_size, n_teams):
    # Initialize lists to store summary data
    swap_pairs = []
    avg_comfort_scores = []
    average_srs = []
    sr_deviations = []
    costs = []
    comfort_components = []
    avg_sr_components = []
    sr_deviation_components = []

    # Check the impact of swapping a player
    for i in range(n_players):
        for j in range(i+1, n_players):
            # Swap players i and j
            swapped_identity = identity.value.copy()
            swapped_identity[[i, j]] = swapped_identity[[j, i]]
            
            # Recalculate the cost components
            swapped_is_role = np.array([sum(swapped_identity[:, k+l] for k in range(0, id_n, n_roles)) for l in range(n_roles)]).T
            swapped_comfs = np.sum(np.multiply(role_comfs, swapped_is_role), axis=1)
            swapped_team_total_srs = np.hstack([np.sum(np.multiply(swapped_identity[:, k:k+n_roles], role_srs)) for k in range(0, id_n, n_roles)])
            swapped_team_avg_srs = swapped_team_total_srs / team_size
            swapped_avg_sr = np.sum(swapped_team_avg_srs) / n_teams
            swapped_sr_deviations = swapped_team_avg_srs - swapped_avg_sr
            
            # Calculate the new cost
            swapped_avg_comf = np.sum(swapped_comfs) / (n_teams * team_size)  # Calculate average comfort
            swapped_cost = WEIGHTS["AVG_COMF_WEIGHT"] * swapped_avg_comf  # Use weighted average comfort instead of total comfort
            swapped_cost += WEIGHTS["AVG_SR_WEIGHT"] * swapped_avg_sr
            swapped_cost -= WEIGHTS["SR_DIFF_PENALTY_WEIGHT"] * (np.max(swapped_team_avg_srs) - np.min(swapped_team_avg_srs))
            
            # Store the summary data
            swap_pairs.append((i, j))
            avg_comfort_scores.append(swapped_avg_comf)
            average_srs.append(np.mean(swapped_team_avg_srs))
            sr_deviations.append(np.mean(np.abs(swapped_team_avg_srs - np.mean(swapped_team_avg_srs))))
            costs.append(swapped_cost)
            comfort_components.append(swapped_avg_comf)  # Store average comfort component
            avg_sr_components.append(WEIGHTS["AVG_SR_WEIGHT"] * swapped_avg_sr)
            sr_deviation_components.append(WEIGHTS["SR_DIFF_PENALTY_WEIGHT"] * (np.max(swapped_team_avg_srs) - np.min(swapped_team_avg_srs)))

    # Plot the summary data
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(avg_comfort_scores, label='Average Comfort Score')
    plt.xlabel('Swap Pair Index')
    plt.ylabel('Average Comfort Score')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(average_srs, label='Average SR')
    plt.xlabel('Swap Pair Index')
    plt.ylabel('Average SR')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(sr_deviations, label='SR Deviation')
    plt.xlabel('Swap Pair Index')
    plt.ylabel('SR Deviation')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(costs, label='Total Cost')
    plt.plot(comfort_components, label=f'Comfort Component (Weight: {WEIGHTS["AVG_COMF_WEIGHT"]})')
    plt.plot(avg_sr_components, label=f'Average SR Component (Weight: {WEIGHTS["AVG_SR_WEIGHT"]})')
    plt.plot(sr_deviation_components, label=f'SR Deviation Component (Weight: {WEIGHTS["SR_DIFF_PENALTY_WEIGHT"]})')
    plt.xlabel('Swap Pair Index')
    plt.ylabel('Cost Components')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary of cost factors
    print("\nSummary of Cost Factors:")
    print(f"Average comfort scores: {avg_comfort_scores}")
    print(f"Average SRs: {average_srs}")
    print(f"SR deviations: {sr_deviations}")
    print(f"Total costs: {costs}")
    print(f"Comfort components: {comfort_components}")
    print(f"Average SR components: {avg_sr_components}")
    print(f"SR deviation components: {sr_deviation_components}")

def team_matching(df):
    # df['Playing'] = df['Playing'].fillna(False)
    # df = df[df['Playing']]
    df.dropna(inplace=True)
    role_srs = df[['Tank SR', 'DPS SR', 'Support SR']].to_numpy()
    role_comfs = df[['Wants Tank', 'Wants Dps', 'Wants Support']].to_numpy()
    ids_df = df[['bnet', 'discord']]
    result_df, avg_srs, devs, comfs, identity, id_n, n_roles, n_players = optimize_team_matching(role_srs, role_comfs, ids_df)
    return result_df, avg_srs, devs, comfs, identity, id_n, n_roles, n_players, role_comfs, role_srs

def main():
    data_path = '5v5 - RanksData (1).csv'
    df, role_srs, role_comfs, ids_df = read_and_preprocess_data(data_path)
    df, avg_srs, devs, comfs, identity, id_n, n_roles, n_players = optimize_team_matching(role_srs, role_comfs, ids_df)
    save_results(df)
    plot_summary_data(n_players, identity, role_comfs, role_srs, WEIGHTS["AVG_SR_WEIGHT"], WEIGHTS["SR_DIFF_PENALTY_WEIGHT"], id_n, n_roles, 5, 2)

if __name__ == "__main__":
    main()
