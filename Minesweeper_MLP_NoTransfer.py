# Minesweeper DQN (MLP agent) with Progressive Training, Per-Stage Exploration, Reward Tracking, and Evaluation

import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import csv
import time

# --- Training Stages Configuration ---
TRAINING_STAGES = [
    {"episodes": 10000, "rows": 6, "cols": 6, "mines": 3, "epsilon_start": 1.0, "epsilon_decay": 0.999},
    {"episodes": 10000,  "rows": 6, "cols": 6, "mines": 4, "epsilon_start": 0.5, "epsilon_decay": 0.9995},
    {"episodes": 10000,  "rows": 6, "cols": 6, "mines": 5, "epsilon_start": 0.5, "epsilon_decay": 0.9995},
]

RUN_TITLE = 'MLP_without_transfer_3stage_6x6_345_30000epi_s2epst05epdc09995'

EVAL_EPISODES = 1000
MAX_STEPS_MULTIPLIER = 2
LEARNING_RATE = 0.001
DISCOUNT = 0.95
MIN_EPSILON = 0.01
BATCH_SIZE = 128
SAVE_MODEL_EVERY = 100
PRINT_RATE = 200


class MinesweeperGame:
    def __init__(self, rows, cols, num_mines):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reset()

    def reset(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.revealed = [[False]*self.cols for _ in range(self.rows)]
        self.flags = [[False]*self.cols for _ in range(self.rows)]
        self.game_over = False
        self.win = False
        self.first_click = True

    def place_mines(self, r0, c0):
        safe_zone = {(r0+i, c0+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if 0 <= r0+i < self.rows and 0 <= c0+j < self.cols}
        placed = 0
        while placed < self.num_mines:
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) not in safe_zone and self.board[r][c] != -1:
                self.board[r][c] = -1
                placed += 1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:
                    continue
                self.board[r][c] = sum(
                    self.board[r+dr][c+dc] == -1
                    for dr in [-1,0,1] for dc in [-1,0,1]
                    if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols
                )

    def reveal(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.revealed[r][c] or self.flags[r][c]:
            return 0, False
        if self.first_click:
            self.place_mines(r, c)
            self.first_click = False
        self.revealed[r][c] = True
        if self.board[r][c] == -1:
            self.game_over, self.win = True, False
            return -10, True
        reward = 1 if self.board[r][c] > 0 else 5
        if self.board[r][c] == 0:
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not self.revealed[nr][nc]:
                            r2, _ = self.reveal(nr, nc)
                            reward += r2
        self._check_win()
        if self.win:
            return reward + 50, True
        return reward, False

    def _check_win(self):
        count = sum(self.revealed[r][c] for r in range(self.rows) for c in range(self.cols) if self.board[r][c] != -1)
        if count == self.rows * self.cols - self.num_mines:
            self.game_over, self.win = True, True

    def get_state(self):
        state = np.zeros((self.rows, self.cols, 3))
        for r in range(self.rows):
            for c in range(self.cols):
                state[r,c,0] = self.revealed[r][c]
                state[r,c,1] = self.flags[r][c]
                state[r,c,2] = self.board[r][c] if self.revealed[r][c] else -2
        return state.flatten().reshape(1, -1)

    def get_valid_actions(self):
        return [(r,c) for r in range(self.rows) for c in range(self.cols) if not self.revealed[r][c] and not self.flags[r][c]]


class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, x, y_true, y_pred, lr):
        dL = y_pred - y_true
        dW2 = self.a1.T @ dL
        db2 = np.sum(dL, axis=0, keepdims=True)
        da1 = dL @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        for grad in [dW1, db1, dW2, db2]:
            np.clip(grad, -1.0, 1.0, out=grad)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def copy(self):
        clone = SimpleMLP(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1])
        clone.W1 = self.W1.copy()
        clone.b1 = self.b1.copy()
        clone.W2 = self.W2.copy()
        clone.b2 = self.b2.copy()
        return clone


class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.q_net = SimpleMLP(input_dim, 128, output_dim)
        self.target_net = self.q_net.copy()
        self.buffer = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999

    def act(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = self.q_net.forward(state)[0]
        scores = [q_vals[r * 6 + c] for r, c in valid_actions]
        return valid_actions[int(np.argmax(scores))]

    def remember(self, *transition):
        self.buffer.append(transition)

    def replay(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.buffer, BATCH_SIZE)
        states, targets = [], []
        for s, a, r, s2, done in batch:
            r = np.clip(r, -10, 50)
            q = self.q_net.forward(s)
            q_next = self.target_net.forward(s2)
            idx = a[0] * 6 + a[1]
            target_val = r if done else r + DISCOUNT * np.max(q_next)
            q[0, idx] = np.clip(target_val, -100, 100)
            states.append(s)
            targets.append(q)
        states = np.vstack(states)
        targets = np.vstack(targets)
        self.q_net.backward(states, targets, self.q_net.forward(states), LEARNING_RATE)

    def update_target(self):
        self.target_net = self.q_net.copy()

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * self.epsilon_decay)


# --- Visualization Function ---
def visualize_game(steps, board, title="Replay"):
    rows, cols = len(board), len(board[0])
    fig, axs = plt.subplots(1, len(steps), figsize=(len(steps) * 2, 5))
    if len(steps) == 1:
        axs = [axs]
    for ax, (step_num, (r, c), reward, revealed) in zip(axs, steps):
        img = np.zeros((rows, cols))
        for y in range(rows):
            for x in range(cols):
                img[y, x] = board[y][x] if revealed[y][x] else -2
        ax.imshow(img, cmap="Blues", vmin=-2, vmax=8)
        for y in range(rows):
            for x in range(cols):
                if revealed[y][x]:
                    ax.text(x, y, str(board[y][x]), ha='center', va='center', color='black', fontsize=12)
        ax.set_title(f"Step {step_num}\n({r},{c})\nR: {reward}")
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


# --- Plot save function ---
def save_results_csv(win_rates, reward_avgs, stage_switches, last_eval_rate, tag="cnn_with_transfer"):
    """Save training results to CSV files for comparison later."""

    # Save win rates per PRINT_RATE episodes
    with open(f"results/win_rates_{tag}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "WinRate"])
        for i, wr in enumerate(win_rates):
            writer.writerow([i * PRINT_RATE, wr])

    # Save average rewards (same length as win_rates)
    with open(f"results/rewards_{tag}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "AvgReward"])
        for i, r in enumerate(reward_avgs):
            writer.writerow([i * PRINT_RATE, r])

    # Save evaluation summary
    with open("results/eval_summary.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # file is empty
            writer.writerow(["ModelTag", "EvalWinRate"])
        writer.writerow([tag, last_eval_rate])


# --- Result CSV Saving Functions ---
def save_winrate_data_csv(win_rates_all, stage_switches, print_rate, tag="run"):
    with open(f"results/{tag}_wrdpt.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "WinRate"])
        for i, wr in enumerate(win_rates_all):
            episode = i * print_rate
            writer.writerow([episode, wr])
        writer.writerow([])
        writer.writerow(["StageSwitchIndexes"])
        writer.writerow(stage_switches)
    print('winrate save succesful')

def save_train_vs_eval_csv(train_win_rate, eval_win_rate, tag="run"):
    with open(f"results/{tag}_tvewr.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "WinRate"])
        writer.writerow(["Train", train_win_rate])
        writer.writerow(["Eval", eval_win_rate])
    print('train vs eval save succesful')

def save_game_steps_csv(steps, board, tag, suffix):
    with open(f"results/{tag}_{suffix}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Action", "Reward", "RevealedBoard"])
        for step_num, action, reward, revealed in steps:
            writer.writerow([step_num, action, reward, str(revealed)])
        writer.writerow([])
        writer.writerow(["Board"])
        for row in board:
            writer.writerow(row)
    print('game visualization save succesful')

# --- Training Function ---
def train():
    start_time = time.time()  # Start timer
    input_dim = 6 * 6 * 3
    output_dim = 6 * 6
    agent = DQNAgent(input_dim, output_dim)
    win_rates_all, reward_avgs_all, stage_switches = [], [], []
    first_game, last_game = None, None

    for stage_idx, stage in enumerate(TRAINING_STAGES):
        wins, rewards, win_rates = 0, [], []
        game = MinesweeperGame(stage['rows'], stage['cols'], stage['mines'])
        agent.epsilon = stage['epsilon_start']
        agent.epsilon_decay = stage['epsilon_decay']

        for ep in range(1, stage['episodes'] + 1):
            game.reset()
            s = game.get_state()
            total_reward, steps = 0, []

            for step in range(stage['rows'] * stage['cols'] * MAX_STEPS_MULTIPLIER):
                actions = game.get_valid_actions()
                if not actions: break
                a = agent.act(s, actions)
                r, done = game.reveal(*a)
                steps.append((step + 1, a, r, [row[:] for row in game.revealed]))
                s2 = game.get_state()
                agent.remember(s, a, r, s2, done)
                agent.replay()
                s = s2
                total_reward += r
                if done: break

            if ep == 1 and stage_idx == 0:
                first_game = (steps, [row[:] for row in game.board])
            if ep == stage['episodes']:
                last_game = (steps, [row[:] for row in game.board])

            if game.win: wins += 1
            rewards.append(total_reward)
            agent.decay_epsilon()

            if ep % PRINT_RATE == 0:
                rate = wins / ep
                win_rates.append(rate)
                avg_reward = np.mean(rewards[-PRINT_RATE:])
                reward_avgs_all.append(avg_reward)
                print(f"Stage {stage_idx+1} Ep {ep}, Win Rate: {rate:.3f}, Avg Reward: {avg_reward:.1f}, Eps: {agent.epsilon:.3f}")

            if ep % SAVE_MODEL_EVERY == 0:
                agent.update_target()

        win_rates_all.extend(win_rates)
        stage_switches.append(len(win_rates_all))

    # Win rate plot with stage switches
    plt.plot(np.arange(len(win_rates_all)) * PRINT_RATE, win_rates_all, label="Win Rate")
    for x in [s * PRINT_RATE for s in stage_switches[:-1]]:
        plt.axvline(x=x, color='red', linestyle='--', alpha=0.6)
    plt.title("Win Rate During Progressive Training")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.show()

    # Evaluation
    game = MinesweeperGame(6, 6, 5)
    agent.epsilon = 0.0
    eval_wins, eval_game = 0, None

    for ep in range(EVAL_EPISODES):
        game.reset()
        s = game.get_state()
        steps = []
        for step in range(6 * 6 * MAX_STEPS_MULTIPLIER):
            actions = game.get_valid_actions()
            if not actions: break
            a = agent.act(s, actions)
            r, done = game.reveal(*a)
            steps.append((step + 1, a, r, [row[:] for row in game.revealed]))
            s = game.get_state()
            if done: break
        if ep == EVAL_EPISODES - 1:
            eval_game = (steps, [row[:] for row in game.board])
        if game.win: eval_wins += 1

    # Compute final stage training win rate for comparison
    last_stage_win_rates = win_rates_all[stage_switches[-2]:]
    eval_win_rate = eval_wins / EVAL_EPISODES

    print(f"\nTraining win rate: {sum(win_rates_all)/len(win_rates_all):.3f}")
    print(f"Last stage training win rate: {np.mean(last_stage_win_rates):.3f}")
    print(f"Evaluation win rate: {eval_wins / EVAL_EPISODES:.3f}")
    
    plt.bar(["Training", "Evaluation"], [sum(win_rates_all)/len(win_rates_all), eval_wins / EVAL_EPISODES], color=['skyblue', 'orange'])
    plt.ylabel("Win Rate")
    plt.title("Training vs Evaluation Win Rate")
    plt.grid(True, axis='y')
    plt.ylim(0, 1)
    plt.show()

    duration = time.time() - start_time
    print(f"\nTotal training time: {duration:.2f} seconds")


    # Save CSV data
    save_results_csv(win_rates_all, reward_avgs_all, stage_switches, eval_win_rate, tag=RUN_TITLE)

    # Save Win Rate During Progressive Training (WRDPT)
    save_winrate_data_csv(win_rates_all, stage_switches, PRINT_RATE, tag=RUN_TITLE)

    # Save Training vs Evaluation Win Rate (TVEWR)
    last_stage_win_rates = win_rates_all[stage_switches[-2]:]
    train_win_rate = np.mean(last_stage_win_rates)
    save_train_vs_eval_csv(train_win_rate, eval_win_rate, tag=RUN_TITLE)

    # Save First, Last, and Final Game Step Data
    save_game_steps_csv(first_game[0], first_game[1], RUN_TITLE, "ftg")
    save_game_steps_csv(last_game[0], last_game[1], RUN_TITLE, "ltg")
    save_game_steps_csv(eval_game[0], eval_game[1], RUN_TITLE, "feg")



    return first_game, last_game, eval_game


if __name__ == "__main__":
    first, last, final = train()
    print("\nVisualizing First Training Game:")
    visualize_game(*first, title="First Training Game")
    print("\nVisualizing Last Training Game:")
    visualize_game(*last, title="Last Training Game")
    print("\nVisualizing Final Evaluation Game:")
    visualize_game(*final, title="Final Evaluation Game")