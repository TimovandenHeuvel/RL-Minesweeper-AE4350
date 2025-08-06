# Improved Minesweeper with Evaluation After Training

import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

# --- Hyperparameters ---
BOARD_ROWS, BOARD_COLS, NUM_MINES = 6, 6, 3
NUM_EPISODES = 30000
EVAL_EPISODES = 1000
MAX_STEPS = BOARD_ROWS * BOARD_COLS * 2
LEARNING_RATE = 0.001
DISCOUNT = 0.95
EPSILON_START, EPSILON_DECAY, MIN_EPSILON = 1.0, 0.999, 0.01
BATCH_SIZE = 128
SAVE_MODEL_EVERY = 100
print_rate = 500


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
        self.epsilon = EPSILON_START

    def act(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = self.q_net.forward(state)[0]
        scores = [q_vals[r * BOARD_COLS + c] for r, c in valid_actions]
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
            idx = a[0] * BOARD_COLS + a[1]
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
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


# --- Visualization Function ---
def visualize_game(steps, board, title="Replay", save_path=None):
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
    if save_path:
        plt.savefig(save_path)
    plt.show()


# --- Training + Evaluation ---
def train():
    game = MinesweeperGame(BOARD_ROWS, BOARD_COLS, NUM_MINES)
    agent = DQNAgent(BOARD_ROWS * BOARD_COLS * 3, BOARD_ROWS * BOARD_COLS)
    win_rates = []
    total_wins = 0
    first_game_steps, first_game_board = [], []
    last_game_steps, last_game_board = [], []
    final_game_steps, final_game_board = [], []

    for ep in range(1, NUM_EPISODES + 1):
        game.reset()
        s = game.get_state()
        total_reward = 0
        steps = []

        for step in range(MAX_STEPS):
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

        if ep == 1:
            first_game_steps = steps
            first_game_board = [row[:] for row in game.board]
        if ep == NUM_EPISODES:
            last_game_steps = steps
            last_game_board = [row[:] for row in game.board]

        if game.win: total_wins += 1
        agent.decay_epsilon()
        if ep % print_rate == 0:
            win_rate = total_wins / ep
            win_rates.append(win_rate)
            print(f"Ep {ep}, Game Won: {game.win}, Win Rate: {win_rate:.2f}, Eps: {agent.epsilon:.2f}, Reward: {total_reward:.1f}")
            q_vals = agent.q_net.forward(s)[0]
            print(f"Q mean: {np.mean(q_vals):.2f}, max: {np.max(q_vals):.2f}, min: {np.min(q_vals):.2f}")

        if ep % 100 == 0:
            with open("model.pkl", "wb") as f:
                pickle.dump(agent.q_net, f)
            agent.update_target()

    plt.plot([i*print_rate for i in range(len(win_rates))], win_rates)
    plt.title("Win Rate During Training")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.show()

    # Evaluation phase (epsilon = 0)
    agent.epsilon = 0.0
    eval_wins = 0
    for eval_ep in range(EVAL_EPISODES):
        game.reset()
        s = game.get_state()
        steps = []
        for step in range(MAX_STEPS):
            actions = game.get_valid_actions()
            if not actions: break
            a = agent.act(s, actions)
            r, done = game.reveal(*a)
            steps.append((step + 1, a, r, [row[:] for row in game.revealed]))
            s = game.get_state()
            if done: break
        if eval_ep == EVAL_EPISODES - 1:
            final_game_steps = steps
            final_game_board = [row[:] for row in game.board]
        if game.win:
            eval_wins += 1

    eval_win_rate = eval_wins / EVAL_EPISODES
    train_win_rate = total_wins / NUM_EPISODES
    print(f"\nTraining win rate (episodes 1–{NUM_EPISODES}): {train_win_rate:.3f}")
    print(f"Evaluation win rate (episodes {NUM_EPISODES+1}–{NUM_EPISODES+EVAL_EPISODES}): {eval_win_rate:.3f}")

    # Plot evaluation win rate
    plt.bar(["Training", "Evaluation"], [train_win_rate, eval_win_rate], color=['skyblue', 'orange'])
    plt.ylabel("Win Rate")
    plt.title("Training vs Evaluation Performance")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.show()

    return agent, (first_game_steps, first_game_board), (last_game_steps, last_game_board), (final_game_steps, final_game_board)


if __name__ == "__main__":
    agent, (first_steps, first_board), (last_steps, last_board), (final_steps, final_board) = train()
    print("\nVisualizing First Game:")
    visualize_game(first_steps, first_board, title="First Game Replay")
    print("\nVisualizing Last Training Game:")
    visualize_game(last_steps, last_board, title="Last Game Replay")
    print("\nVisualizing Final Evaluation Game:")
    visualize_game(final_steps, final_board, title="Final Eval Game Replay")
