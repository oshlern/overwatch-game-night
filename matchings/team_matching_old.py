import pandas as pd
import numpy as np
import cvxpy as cp

# Read the data file
data_path = '5v5 - RanksData (1).csv'
df = pd.read_csv(data_path)
# df.head()

df = df[df['Playing'].notna()]
df.dropna(inplace=True)
print(df)

role_srs = df[['Tank SR', 'DPS SR', 'Support SR']].to_numpy()# * 4
role_comfs = df[['Wants Tank', 'Wants Dps', 'Wants Support']].to_numpy()
df.rename(columns={'bnet': 'Overwatch', 'discord': 'Discord'}, inplace=True)

ids_df = df[['Overwatch', 'Discord']]

n_teams = 2
n_roles = 3
team_size = 5
# n = n_teams * team_size
n_players = len(ids_df)


max_sr_diff = 5
# avg_sr_weight = 2/5000
avg_sr_weight = 0.005

id_n = n_teams * n_roles
identity = cp.Variable((n_players, id_n), boolean=True)
constraints = [cp.sum(identity, axis=1) <= 1] # could have someone not play

team_roles = cp.sum(identity, axis=0) # n_teams x n_roles
for i in range(n_teams):
    constraints += [team_roles[3*i] == 1]
    constraints += [team_roles[3*i+1] == 2]
    constraints += [team_roles[3*i+2] == 2]
# constraints += [team_roles == 2]

is_role = cp.vstack([sum(identity[:,i+j] for i in range(0,id_n,n_roles)) for j in range(n_roles)]).T
comfs = cp.sum(cp.multiply(role_comfs, is_role), axis=1)
constraints += [comfs >= 0]

team_total_srs = cp.hstack([cp.sum(cp.multiply(identity[:,i:i+n_roles],role_srs)) for i in range(0,id_n,n_roles)])
team_avg_srs = team_total_srs / team_size
avg_sr = cp.sum(team_avg_srs) / n_teams
sr_deviations = team_avg_srs - avg_sr
constraints += [sr_deviations <= max_sr_diff, sr_deviations >= -max_sr_diff]
# print("team total srs", team_total_srs.shape)
# print("team avg srs", team_avg_srs.shape)
# print("avg sr", avg_sr.shape)
# print("sr deviations", sr_deviations.shape)

# duos = [[8,9], [15, 18], [21, 3]]
# is_team = cp.vstack([sum(identity[:,i+j] for j in range(n_roles)) for i in range(0,id_n,n_roles)]).T
# team_assignments = {i: duos[i] for i in range(len(duos))}
# for i in range(len(duos)):
#     constraints += [is_team[player, i%n_teams] == 1 for player in duos[i]]

cost = cp.sum(comfs) # maximize comfortability
cost += avg_sr_weight * avg_sr #
cost -= 1/10 * (cp.max(team_avg_srs) - cp.min(team_avg_srs))
# cost += affinity
# cost
objective = cp.Maximize(cost)

prob = cp.Problem(objective, constraints)
prob.solve()

ids = identity.value
is_role = np.array([sum(ids[:,i+j] for i in range(0,id_n,n_roles)) for j in range(n_roles)]).T
is_team = np.array([sum(ids[:,i+j] for j in range(n_roles)) for i in range(0,id_n,n_roles)]).T
# teams = np.argmax(is_team, axis=1)
# roles = np.argmax(is_role, axis=1)
is_role = np.rint(is_role).astype(int)
is_team = np.rint(is_team).astype(int)

print("is_team", is_team)
print("is_role", is_role)

teams = np.where(np.all(is_team == 0, axis=1), -1, np.argmax(is_team, axis=1))
roles = np.where(np.all(is_role == 0, axis=1), -1, np.argmax(is_role, axis=1))

print(teams, roles)
srs = np.squeeze(np.take_along_axis(role_srs, roles[:,None], axis=1))
comfs = np.squeeze(np.take_along_axis(role_comfs, roles[:,None], axis=1))
print(srs, comfs)
avg_srs = [np.mean(srs[teams==i]) for i in range(n_teams)]
devs = np.abs(avg_srs - np.mean(srs))
print(np.abs(avg_srs - np.mean(srs)))
print(f"comf: {np.mean(comfs)}, sr: {np.mean(avg_srs)} +- {np.mean(devs)}")
print(np.max(avg_srs), np.min(avg_srs))
print(np.std(avg_srs))


# Create a DataFrame with the matchings
df = pd.DataFrame({
    "Battletag": ids_df["Overwatch"],
    "Discord": ids_df["Discord"],
    "SR": role_srs.max(axis=1),  # Example SR values
    "Comf": role_comfs.max(axis=1),  # Example comfort values
    "Role": [roles[i] for i in range(len(roles))],
    "Team": [teams[i] for i in range(len(teams))]
}).sort_values(["Team", "Role"])

# Remove rows where teams is -1
df = df[df["Team"] != -1]

# Save the DataFrame to a CSV file
df.to_csv("matchings.csv", index=False)
# Group by teams and create a list of battletags for each team
grouped_teams = df.groupby("Team")["Battletag"].apply(lambda x: '\t'.join(x)).reset_index()

# Save only the battletags to a txt file with a row per team, tab separated values, no headers or indices
grouped_teams["Battletag"].to_csv("grouped_teams.txt", index=False, header=False, sep='\n')
