import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dictionary for training win rate files
files = {
    "Learning Rate (0.0001) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 1/Dc0.95_Lr0.0001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.0001) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 2/Dc0.95_Lr0.0001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.0001) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 3/Dc0.95_Lr0.0001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.0001) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 4/Dc0.95_Lr0.0001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.0001) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 5/Dc0.95_Lr0.0001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",

    "Learning Rate (0.001) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.001) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.001) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.001) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.001) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    
    "Learning Rate (0.01) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 1/Dc0.95_Lr0.01_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.01) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 2/Dc0.95_Lr0.01_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.01) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 3/Dc0.95_Lr0.01_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.01) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 4/Dc0.95_Lr0.01_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "Learning Rate (0.01) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 5/Dc0.95_Lr0.01_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    }

# Dictionary for evaluation win rate files
eval_files = {
    "Learning Rate (0.0001) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 1/Dc0.95_Lr0.0001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.0001) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 2/Dc0.95_Lr0.0001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.0001) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 3/Dc0.95_Lr0.0001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.0001) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 4/Dc0.95_Lr0.0001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.0001) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.0001/Run 5/Dc0.95_Lr0.0001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",

    "Learning Rate (0.001) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.001) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.001) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.001) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.001) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.001/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    
    "Learning Rate (0.01) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 1/Dc0.95_Lr0.01_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.01) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 2/Dc0.95_Lr0.01_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.01) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 3/Dc0.95_Lr0.01_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.01) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 4/Dc0.95_Lr0.01_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    "Learning Rate (0.01) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Learning Rate/0.01/Run 5/Dc0.95_Lr0.01_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_tvewr.csv",
    }


# -----------------------------------------------------------
# Read all training files and group them by Learning Rate rate
# -----------------------------------------------------------

# Lists to hold the dataframes for each Learning Rate rate
lrs_0_0001 = []
lrs_0_001 = []
lrs_0_01 = []

for label, path in files.items():
    try:
        # Read the CSV file, skipping the last two rows as they are footers
        df = pd.read_csv(path, skipfooter=2, engine='python')
        
        # Determine which group the file belongs to and append it
        if "Learning Rate (0.0001)" in label:
            lrs_0_0001.append(df)
        elif "Learning Rate (0.001)" in label:
            lrs_0_001.append(df)
        elif "Learning Rate (0.01)" in label:
            lrs_0_01.append(df)
            
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")

# -----------------------------------------------------------
# Calculate the average win rate for each training group
# -----------------------------------------------------------

# Concatenate all dataframes for the 0.0001 Learning Rate group
combined_0_0001_lr = pd.concat(lrs_0_0001)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_0001 = combined_0_0001_lr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_0001 = combined_0_0001_lr.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.001 Learning Rate group
combined_0_001_lr = pd.concat(lrs_0_001)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_001 = combined_0_001_lr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_001 = combined_0_001_lr.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.01 Learning Rate group
combined_0_01_lr = pd.concat(lrs_0_01)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_01 = combined_0_01_lr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_01 = combined_0_01_lr.groupby('Episode')['WinRate'].std()



# Print the average winrates and their standard deviation for specific episodes
print("\nAverage Winrate for Learning Rate (0.0001) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_0001.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Learning Rate (0.0001) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_0001.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Learning Rate (0.001) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_001.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Learning Rate (0.001) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_001.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Learning Rate (0.01) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_01.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Learning Rate (0.01) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_01.loc[[9800, 19800, 29800]])


# -----------------------------------------------------------
# Read and process evaluation files
# -----------------------------------------------------------
eval_winrates = {
    '0.0001': [],
    '0.001': [],
    '0.01': [],
}

for label, path in eval_files.items():
    try:
        df = pd.read_csv(path)
        eval_winrate = df[df['Metric'] == 'Eval']['WinRate'].values[0]
        if "Learning Rate (0.0001)" in label:
            eval_winrates['0.0001'].append(eval_winrate)
        elif "Learning Rate (0.001)" in label:
            eval_winrates['0.001'].append(eval_winrate)
        elif "Learning Rate (0.01)" in label:
            eval_winrates['0.01'].append(eval_winrate)
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
plt.plot(avg_winrate_0_0001.index, avg_winrate_0_0001.values, label="Average Learning Rate (0.0001)")
plt.plot(avg_winrate_0_001.index, avg_winrate_0_001.values, label="Average Learning Rate (0.001)")
plt.plot(avg_winrate_0_01.index, avg_winrate_0_01.values, label="Average Learning Rate (0.01)")

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
Learning_Rates = list(avg_eval_winrates.keys())
avg_wins = list(avg_eval_winrates.values())
std_devs = list(std_eval_winrates.values())

plt.bar(Learning_Rates, avg_wins, yerr=std_devs, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylabel("Win Rate")
plt.xlabel("Learning Rate")
plt.title("Evaluation Win Rate Comparison")
plt.grid(True, axis='y')
plt.ylim(0, 1)
plt.show()