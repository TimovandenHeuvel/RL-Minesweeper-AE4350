import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time  # For tracking total runtime


# --- Hyperparameters ---
TRAINING_STAGES = [
    {"episodes": 10000, "rows": 6, "cols": 6, "mines": 3, "epsilon_start": 1.0, "epsilon_decay": 0.999},
    {"episodes": 10000, "rows": 6, "cols": 6, "mines": 4, "epsilon_start": 0.5, "epsilon_decay": 0.9995},
    {"episodes": 10000, "rows": 6, "cols": 6, "mines": 5, "epsilon_start": 0.5, "epsilon_decay": 0.9995},
]


EVAL_EPISODES = 1000                  # Number of episodes during final evaluation
MAX_STEPS_MULTIPLIER = 2              # Multiplies board size to cap steps
LEARNING_RATE = 0.001                 # Adam optimizer learning rate
DISCOUNT = 0.95                       # Discount factor (gamma) for future rewards
MIN_EPSILON = 0.01                    # Lower bound for exploration
BATCH_SIZE = 128                      # Experience replay batch size
SAVE_MODEL_EVERY = 100                # Sync target network every N episodes
PRINT_RATE = 500                      # How often to print performance info

# --- Minesweeper Environment ---
class MinesweeperGame:
    """A simplified Minesweeper game environment."""

    def __init__(self, rows, cols, num_mines):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reset()

    def reset(self):
        """Reset the game board and internal state."""
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.revealed = [[False]*self.cols for _ in range(self.rows)]
        self.game_over = False
        self.win = False
        self.first_click = True

    def place_mines(self, r0, c0):
        """Place mines randomly, avoiding the 3x3 zone around first click."""
        safe_zone = {(r0+i, c0+j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                     if 0 <= r0+i < self.rows and 0 <= c0+j < self.cols}
        placed = 0
        while placed < self.num_mines:
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) not in safe_zone and self.board[r][c] != -1:
                self.board[r][c] = -1
                placed += 1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1: continue
                self.board[r][c] = sum(
                    self.board[r+dr][c+dc] == -1
                    for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                    if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols
                )

    def reveal(self, r, c):
        """Reveal a cell, return reward and whether game ended."""
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.revealed[r][c]:
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
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not self.revealed[nr][nc]:
                            r2, _ = self.reveal(nr, nc)
                            reward += r2
        self._check_win()
        if self.win: return reward + 50, True
        return reward, False

    def _check_win(self):
        """Check if all safe tiles are revealed."""
        count = sum(self.revealed[r][c] for r in range(self.rows)
                    for c in range(self.cols) if self.board[r][c] != -1)
        if count == self.rows * self.cols - self.num_mines:
            self.game_over, self.win = True, True

    def get_state(self):
        """Return current game state as a 2xRxC tensor."""
        state = np.zeros((2, self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                state[0, r, c] = self.revealed[r][c]
                state[1, r, c] = self.board[r][c] if self.revealed[r][c] else -2
        return state

    def get_valid_actions(self):
        """Return list of unrevealed cell coordinates."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if not self.revealed[r][c]]

# --- Convolutional Q-Network ---
class SimpleCNN(nn.Module):
    """CNN model to predict Q-values for each cell."""

    def __init__(self, input_channels, rows, cols):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * rows * cols, 256), nn.ReLU(),
            nn.Linear(256, rows * cols)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# --- DQN Agent ---
class DQNAgent:
    """Reinforcement learning agent using Deep Q-Learning."""

    def __init__(self, rows, cols, epsilon_start=1.0, epsilon_decay=0.999):
        self.rows, self.cols = rows, cols
        self.q_net = SimpleCNN(2, rows, cols)
        self.target_net = SimpleCNN(2, rows, cols)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.buffer = deque(maxlen=10000)
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay

    def act(self, state, valid_actions):
        """Select action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.tensor(state.reshape(1, 2, self.rows, self.cols), dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.q_net(state_tensor).cpu().numpy()[0]
        scores = [q_vals[r * self.cols + c] for r, c in valid_actions]
        return valid_actions[int(np.argmax(scores))]

    def remember(self, *transition):
        self.buffer.append(transition)

    def replay(self):
        """Train the Q-network from experience replay buffer."""
        if len(self.buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.buffer, BATCH_SIZE)
        states, targets = [], []

        for s, a, r, s2, done in batch:
            r = np.clip(r, -10, 50)
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            s2_tensor = torch.tensor(s2, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(s_tensor)
            with torch.no_grad():
                next_q_values = self.target_net(s2_tensor)
            idx = a[0] * self.cols + a[1]
            target_q = q_values.clone()
            target_val = r if done else r + DISCOUNT * torch.max(next_q_values)
            target_val = max(-100.0, min(100.0, float(target_val)))
            target_q[0, idx] = target_val
            states.append(s_tensor)
            targets.append(target_q)

        states = torch.cat(states)
        targets = torch.cat(targets)
        preds = self.q_net(states)
        loss = self.loss_fn(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Sync target network with Q-network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Apply epsilon decay after each episode."""
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
        
        
# --- Training Function ---
def train():
    start_time = time.time()  # Start timer
    win_rates_all, reward_avgs_all, stage_switches = [], [], []
    first_game, last_game = None, None

    agent = None
    prev_rows, prev_cols = None, None

    for stage_idx, stage in enumerate(TRAINING_STAGES):
        rows, cols = stage['rows'], stage['cols']
        mines = stage['mines']

        # Initialize or adapt the agent
        if agent is None:
            print(f"\nCreating new agent for {rows}x{cols}")
            agent = DQNAgent(rows, cols, stage['epsilon_start'], stage['epsilon_decay'])
        else:
            if rows != prev_rows or cols != prev_cols:
                print(f"\nGrid size changed: {prev_rows}x{prev_cols} → {rows}x{cols}")
                new_agent = DQNAgent(rows, cols, stage['epsilon_start'], stage['epsilon_decay'])

                # Transfer convolutional weights
                new_agent.q_net.conv.load_state_dict(agent.q_net.conv.state_dict())
                new_agent.target_net.conv.load_state_dict(agent.target_net.conv.state_dict())

                # Replace agent (fully connected layers are rebuilt)
                agent = new_agent

        prev_rows, prev_cols = rows, cols
        agent.epsilon = stage['epsilon_start']
        agent.epsilon_decay = stage['epsilon_decay']

        game = MinesweeperGame(rows, cols, mines)
        wins, rewards, win_rates = 0, [], []

        for ep in range(1, stage['episodes'] + 1):
            game.reset()
            s = game.get_state()
            total_reward, steps = 0, []

            for step in range(rows * cols * MAX_STEPS_MULTIPLIER):
                actions = game.get_valid_actions()
                if not actions: break
                a = agent.act(s, actions)
                r, done = game.reveal(*a)
                steps.append((step + 1, a, r, [row[:] for row in game.revealed]))
                s2 = game.get_state()
                agent.remember(s, a, r, s2, done)
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
                
            agent.replay() 
            
        win_rates_all.extend(win_rates)
        stage_switches.append(len(win_rates_all))

    # Plot training win rate
    plt.plot(np.arange(len(win_rates_all)) * PRINT_RATE, win_rates_all, label="Win Rate")
    for x in [s * PRINT_RATE for s in stage_switches[:-1]]:
        plt.axvline(x=x, color='red', linestyle='--', alpha=0.6)
    plt.title("Win Rate During Progressive Training")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.show()

    # Final evaluation on last stage board
    game = MinesweeperGame(prev_rows, prev_cols, stage['mines'])
    agent.epsilon = 0.0
    eval_wins, eval_game = 0, None

    for ep in range(EVAL_EPISODES):
        game.reset()
        s = game.get_state()
        steps = []
        for step in range(prev_rows * prev_cols * MAX_STEPS_MULTIPLIER):
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

    plt.bar(["Last Stage (Train)", "Evaluation"], [np.mean(last_stage_win_rates), eval_wins / EVAL_EPISODES], color=['skyblue', 'orange'])
    plt.ylabel("Win Rate")
    plt.title("Training vs Evaluation Win Rate")
    plt.grid(True, axis='y')
    plt.ylim(0, 1)
    plt.show()

    duration = time.time() - start_time
    print(f"\nTotal training time: {duration:.2f} seconds")
    
    return first_game, last_game, eval_game


# --- Main Script ---
if __name__ == "__main__":
    first, last, final = train()
    visualize_game(*first, title="First Training Game")
    visualize_game(*last, title="Last Training Game")
    visualize_game(*final, title="Final Evaluation Game")
