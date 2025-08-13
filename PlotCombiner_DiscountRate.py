import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dictionary for training win rate files
files = {
    "Discount (0.05) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 1/Dc0.05_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.05) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 2/Dc0.05_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.05) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 3/Dc0.05_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.05) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 4/Dc0.05_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.05) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 5/Dc0.05_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",

    "Discount (0.50) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 1/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.50) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 2/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.50) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 3/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.50) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 4/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.50) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 5/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",

    "Discount (0.95) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.95) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 2/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.95) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 3/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.95) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 4/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Discount (0.95) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 5/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
}

# Dictionary for evaluation win rate files
eval_files = {
    "Discount (0.05) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 1/Dc0.05_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.05) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 2/Dc0.05_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.05) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 3/Dc0.05_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.05) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 4/Dc0.05_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.05) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.05/Run 5/Dc0.05_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",

    "Discount (0.50) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 1/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.50) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 2/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.50) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 3/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.50) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 4/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.50) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.50/Run 5/Dc0.5_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",

    "Discount (0.95) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.95) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 2/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.95) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 3/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.95) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 4/Dc0.95_Lr0.001_Run_0_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Discount (0.95) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Discount/Runs 0.95/Run 5/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
}


# -----------------------------------------------------------
# Read all training files and group them by discount rate
# -----------------------------------------------------------

# Lists to hold the dataframes for each discount rate
dfs_0_05 = []
dfs_0_50 = []
dfs_0_95 = []

for label, path in files.items():
    try:
        # Read the CSV file, skipping the last two rows as they are footers
        df = pd.read_csv(path, skipfooter=2, engine='python')
        
        # Determine which group the file belongs to and append it
        if "Discount (0.05)" in label:
            dfs_0_05.append(df)
        elif "Discount (0.50)" in label:
            dfs_0_50.append(df)
        elif "Discount (0.95)" in label:
            dfs_0_95.append(df)
            
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")

# -----------------------------------------------------------
# Calculate the average win rate for each training group
# -----------------------------------------------------------

# Concatenate all dataframes for the 0.05 discount group
combined_0_05_df = pd.concat(dfs_0_05)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_05 = combined_0_05_df.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_05 = combined_0_05_df.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.50 discount group
combined_0_50_df = pd.concat(dfs_0_50)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_50 = combined_0_50_df.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_50 = combined_0_50_df.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.95 discount group
combined_0_95_df = pd.concat(dfs_0_95)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_95 = combined_0_95_df.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_95 = combined_0_95_df.groupby('Episode')['WinRate'].std()



# Print the average winrates and their standard deviation for specific episodes
print("\nAverage Winrate for Discount (0.05) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_05.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Discount (0.05) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_05.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Discount (0.50) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_50.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Discount (0.50) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_50.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Discount (0.95) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_95.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Discount (0.95) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_95.loc[[9800, 19800, 29800]])


# -----------------------------------------------------------
# Read and process evaluation files
# -----------------------------------------------------------
eval_winrates = {
    '0.05': [],
    '0.50': [],
    '0.95': [],
}

for label, path in eval_files.items():
    try:
        df = pd.read_csv(path)
        eval_winrate = df[df['Metric'] == 'Eval']['WinRate'].values[0]
        if "Discount (0.05)" in label:
            eval_winrates['0.05'].append(eval_winrate)
        elif "Discount (0.50)" in label:
            eval_winrates['0.50'].append(eval_winrate)
        elif "Discount (0.95)" in label:
            eval_winrates['0.95'].append(eval_winrate)
    except FileNotFoundError:
        print(f"Evaluation file not found: {path}")
    except Exception as e:
        print(f"An error occurred while processing evaluation file {path}: {e}")

# Calculate average and standard deviation for evaluation data
avg_eval_winrates = {key: np.mean(val) for key, val in eval_winrates.items()}
std_eval_winrates = {key: np.std(val) for key, val in eval_winrates.items()}

# Print evaluation results to console
print("\n--- Evaluation Results ---")
print("Average Evaluation Win Rates:")
print(avg_eval_winrates)
print("\nStandard Deviation of Evaluation Win Rates:")
print(std_eval_winrates)

# -----------------------------------------------------------
# PLOTTING THE TRAINING AVERAGE LINES
# -----------------------------------------------------------

plt.figure(figsize=(10, 6))

# Plot the average win rate lines
plt.plot(avg_winrate_0_05.index, avg_winrate_0_05.values, label="Average Discount (0.05)")
plt.plot(avg_winrate_0_50.index, avg_winrate_0_50.values, label="Average Discount (0.50)")
plt.plot(avg_winrate_0_95.index, avg_winrate_0_95.values, label="Average Discount (0.95)")

plt.legend(prop={'size': 8})
plt.title("Average Training Win Rate Comparison")
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# PLOTTING THE EVALUATION BAR CHART
# -----------------------------------------------------------

plt.figure(figsize=(8, 6))
discounts = list(avg_eval_winrates.keys())
avg_wins = list(avg_eval_winrates.values())
std_devs = list(std_eval_winrates.values())

plt.bar(discounts, avg_wins, yerr=std_devs, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylabel("Win Rate")
plt.xlabel("Discount Factor ($\gamma$)")
plt.title("Evaluation Win Rate Comparison")
plt.grid(True, axis='y')
plt.ylim(0, 1)
plt.show()