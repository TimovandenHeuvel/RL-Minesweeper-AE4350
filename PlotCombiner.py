import pandas as pd
import matplotlib.pyplot as plt



files = {
    "CNN (Transfer) 1": "results/30000 episodes/Run 1/CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "CNN (Transfer) 2": "results/30000 episodes/Run 2/CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "CNN (Transfer) 3": "results/30000 episodes/Run 3/CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "CNN (Transfer) 4": "results/30000 episodes/Run 4/CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
    "CNN (Transfer) 5": "results/30000 episodes/Run 5/CNN_with_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995_wrdpt.csv",
}


plt.figure()
for label, path in files.items():
    df = pd.read_csv(path, skipfooter=2, engine='python')
    plt.plot(df["Episode"], df["WinRate"], label=label)

plt.legend(prop={'size': 8})
plt.title("Win Rate Comparison")
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.grid(True)
plt.show()