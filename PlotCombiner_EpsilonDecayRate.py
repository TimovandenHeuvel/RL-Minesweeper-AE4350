import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dictionary for training win rate files
files = {
    "Epsilon Decay Rate (0.999) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_wrdpt.csv",
    "Epsilon Decay Rate (0.999) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_wrdpt.csv",
    "Epsilon Decay Rate (0.999) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_wrdpt.csv",
    "Epsilon Decay Rate (0.999) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_wrdpt.csv",
    "Epsilon Decay Rate (0.999) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_wrdpt.csv",

    "Epsilon Decay Rate (0.9995) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_wrdpt.csv",
    "Epsilon Decay Rate (0.9995) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_wrdpt.csv",
    "Epsilon Decay Rate (0.9995) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_wrdpt.csv",
    "Epsilon Decay Rate (0.9995) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_wrdpt.csv",
    "Epsilon Decay Rate (0.9995) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_wrdpt.csv",
    
    "Epsilon Decay Rate (0.9999) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_wrdpt.csv",
    "Epsilon Decay Rate (0.9999) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_wrdpt.csv",
    "Epsilon Decay Rate (0.9999) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_wrdpt.csv",
    "Epsilon Decay Rate (0.9999) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_wrdpt.csv",
    "Epsilon Decay Rate (0.9999) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_wrdpt.csv",
    }

# Dictionary for evaluation win rate files
eval_files = {
    "Epsilon Decay Rate (0.999) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_tvewr.csv",
    "Epsilon Decay Rate (0.999) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_tvewr.csv",
    "Epsilon Decay Rate (0.999) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_tvewr.csv",
    "Epsilon Decay Rate (0.999) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_tvewr.csv",
    "Epsilon Decay Rate (0.999) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.999/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.999_tvewr.csv",

    "Epsilon Decay Rate (0.9995) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_tvewr.csv",
    "Epsilon Decay Rate (0.9995) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_tvewr.csv",
    "Epsilon Decay Rate (0.9995) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_tvewr.csv",
    "Epsilon Decay Rate (0.9995) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_tvewr.csv",
    "Epsilon Decay Rate (0.9995) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9995/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9995_tvewr.csv",
    
    "Epsilon Decay Rate (0.9999) 1": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 1/Dc0.95_Lr0.001_Run_1_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_tvewr.csv",
    "Epsilon Decay Rate (0.9999) 2": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 2/Dc0.95_Lr0.001_Run_2_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_tvewr.csv",
    "Epsilon Decay Rate (0.9999) 3": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 3/Dc0.95_Lr0.001_Run_3_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_tvewr.csv",
    "Epsilon Decay Rate (0.9999) 4": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 4/Dc0.95_Lr0.001_Run_4_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_tvewr.csv",
    "Epsilon Decay Rate (0.9999) 5": "results/30000 episodes/CNN (with transfer)/Sensitivity Analysis/Epsilon Decay Rate/0.9999/Run 5/Dc0.95_Lr0.001_Run_5_CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc0.9999_tvewr.csv",
    }


# -----------------------------------------------------------
# Read all training files and group them by Epsilon Decay Rate
# -----------------------------------------------------------

# Lists to hold the dataframes for each Epsilon Decay Rate
edrs_0_9990 = []
edrs_0_9995 = []
edrs_0_9999 = []

for label, path in files.items():
    try:
        # Read the CSV file, skipping the last two rows as they are footers
        df = pd.read_csv(path, skipfooter=2, engine='python')
        
        # Determine which group the file belongs to and append it
        if "Epsilon Decay Rate (0.999)" in label:
            edrs_0_9990.append(df)
        elif "Epsilon Decay Rate (0.9995)" in label:
            edrs_0_9995.append(df)
        elif "Epsilon Decay Rate (0.9999)" in label:
            edrs_0_9999.append(df)
            
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")

# -----------------------------------------------------------
# Calculate the average win rate for each training group
# -----------------------------------------------------------

# Concatenate all dataframes for the 0.999 Epsilon Decay Rate group
combined_0_9990_edr = pd.concat(edrs_0_9990)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_9990 = combined_0_9990_edr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_9990 = combined_0_9990_edr.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.9995 Epsilon Decay Rate group
combined_0_9995_edr = pd.concat(edrs_0_9995)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_9995 = combined_0_9995_edr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_9995 = combined_0_9995_edr.groupby('Episode')['WinRate'].std()

# Concatenate all dataframes for the 0.9999 Epsilon Decay Rate group
combined_0_9999_edr = pd.concat(edrs_0_9999)
# Calculate the mean 'WinRate' for each 'Episode'
avg_winrate_0_9999 = combined_0_9999_edr.groupby('Episode')['WinRate'].mean()
# Calculate the standard deviation of 'WinRate' for each 'Episode'
std_winrate_0_9999 = combined_0_9999_edr.groupby('Episode')['WinRate'].std()



# Print the average winrates and their standard deviation for specific episodes
print("\nAverage Winrate for Epsilon Decay Rate (0.999) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_9990.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Epsilon Decay Rate (0.999) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_9990.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Epsilon Decay Rate (0.9995) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_9995.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Epsilon Decay Rate (0.9995) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_9995.loc[[9800, 19800, 29800]])

print("\nAverage Winrate for Epsilon Decay Rate (0.9999) runs at episodes 9800, 19800, 29800:")
print(avg_winrate_0_9999.loc[[9800, 19800, 29800]])
print("\nStandard Deviation for Epsilon Decay Rate (0.9999) runs at episodes 9800, 19800, 29800:")
print(std_winrate_0_9999.loc[[9800, 19800, 29800]])


# -----------------------------------------------------------
# Read and process evaluation files
# -----------------------------------------------------------
eval_winrates = {
    '0.999': [],
    '0.9995': [],
    '0.9999': [],
}

for label, path in eval_files.items():
    try:
        df = pd.read_csv(path)
        eval_winrate = df[df['Metric'] == 'Eval']['WinRate'].values[0]
        if "Epsilon Decay Rate (0.999)" in label:
            eval_winrates['0.999'].append(eval_winrate)
        elif "Epsilon Decay Rate (0.9995)" in label:
            eval_winrates['0.9995'].append(eval_winrate)
        elif "Epsilon Decay Rate (0.9999)" in label:
            eval_winrates['0.9999'].append(eval_winrate)
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
plt.plot(avg_winrate_0_9990.index, avg_winrate_0_9990.values, label="Average Epsilon Decay Rate (0.999)")
plt.plot(avg_winrate_0_9995.index, avg_winrate_0_9995.values, label="Average Epsilon Decay Rate (0.9995)")
plt.plot(avg_winrate_0_9999.index, avg_winrate_0_9999.values, label="Average Epsilon Decay Rate (0.9999)")

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
plt.xlabel("Epsilon Decay Rate")
plt.title("Evaluation Win Rate Comparison")
plt.grid(True, axis='y')
plt.ylim(0, 1)
plt.show()