import tkinter as tk
import random
import numpy as np
from collections import deque
import time # For visualization delay

# --- Part 1: Minesweeper Game Implementation ---

class MinesweeperGame:
    """
    Represents the core logic of the Minesweeper game.
    """
    def __init__(self, rows, cols, num_mines):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.board = []  # Stores mine locations (-1) and numbers (0-8)
        self.revealed = [] # True if cell is revealed, False otherwise
        self.flags = []    # True if cell is flagged, False otherwise
        self.game_over = False
        self.win = False
        self.first_click = True # Special handling for the first click

        self._initialize_board()

    def _initialize_board(self):
        """Initializes the board with empty cells, then places mines and calculates numbers."""
        # print("DEBUG: MinesweeperGame._initialize_board() called.") # Debug print - removed for less verbosity
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.revealed = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.game_over = False
        self.win = False
        self.first_click = True

    def place_mines(self, initial_click_row, initial_click_col):
        """
        Places mines randomly on the board, ensuring the initial click location
        and its immediate neighbors are safe.
        """
        mines_placed = 0
        safe_zone = set()
        # Define a 3x3 safe zone around the initial click
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                nr, nc = initial_click_row + r_offset, initial_click_col + c_offset
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    safe_zone.add((nr, nc))

        while mines_placed < self.num_mines:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if self.board[r][c] != -1 and (r, c) not in safe_zone:
                self.board[r][c] = -1  # -1 represents a mine
                mines_placed += 1
        self._calculate_numbers()

    def _calculate_numbers(self):
        """Calculates the number of adjacent mines for each non-mine cell."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:  # It's a mine, skip
                    continue
                
                # Count adjacent mines
                mine_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue # Skip the cell itself
                        
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self.board[nr][nc] == -1:
                                mine_count += 1
                self.board[r][c] = mine_count

    def reveal_cell(self, r, c):
        """
        Reveals a cell. If it's a mine, the game is over. If it's a '0',
        it recursively reveals adjacent cells.
        Returns: reward for the action, and whether the game ended.
        """
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.revealed[r][c] or self.flags[r][c]:
            return 0, False # Invalid move or already revealed/flagged

        if self.first_click:
            self.place_mines(r, c) # Place mines after the first safe click
            self.first_click = False

        self.revealed[r][c] = True
        reward = 1 # Small positive reward for revealing a safe cell

        if self.board[r][c] == -1:
            self.game_over = True
            self.win = False
            reward = -100 # Large negative reward for hitting a mine
            return reward, True
        elif self.board[r][c] == 0:
            # Recursively reveal adjacent cells if it's a 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.revealed[nr][nc] and not self.flags[nr][nc]:
                        # Accumulate reward from recursive reveals
                        recursive_reward, _ = self.reveal_cell(nr, nc)
                        reward += recursive_reward
            reward += 10 # Extra reward for revealing a 0, as it opens up more information

        self._check_win()
        if self.win:
            reward += 100 # Large positive reward for winning
            return reward, True
        
        return reward, self.game_over

    def flag_cell(self, r, c):
        """Toggles a flag on a cell."""
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.revealed[r][c]:
            return 0, False # Cannot flag revealed cells or out of bounds

        current_flag_state = self.flags[r][c]
        self.flags[r][c] = not self.flags[r][c]

        # NEW: Penalize flagging an already flagged cell, or unflagging a non-flagged cell
        if current_flag_state == self.flags[r][c]: # State didn't change (shouldn't happen with `not`)
             # This case implies a logic error or attempting to flag a revealed cell which is already handled
            return 0, False
        elif current_flag_state: # Was flagged, now unflagged
            return -0.5, False # Small penalty for unflagging (could be a mistake)
        else: # Was unflagged, now flagged
            # No immediate reward, but if the agent keeps flagging the same cell,
            # it will get stuck and eventually lose, which is a large negative reward.
            # A small positive reward for *correctly* flagging a mine could be added later,
            # but requires knowing mine locations, which the agent doesn't directly.
            return 0, False # No immediate reward for flagging

    def _check_win(self):
        """Checks if the game has been won."""
        # Win condition: all non-mine cells are revealed
        revealed_safe_cells = 0
        total_safe_cells = (self.rows * self.cols) - self.num_mines
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r][c] and self.board[r][c] != -1:
                    revealed_safe_cells += 1
        
        if revealed_safe_cells == total_safe_cells:
            self.win = True
            self.game_over = True

    def reset(self):
        """Resets the game for a new episode."""
        self._initialize_board()

    def get_hidden_cells_count(self):
        """Returns the number of hidden cells (not revealed and not flagged)."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.revealed[r][c] and not self.flags[r][c]:
                    count += 1
        return count

# --- Part 2: Reinforcement Learning Agent Implementation ---

class MLP:
    """
    A simple Multi-Layer Perceptron (Neural Network) implemented with NumPy.
    Used as a function approximator for Q-values.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate, clip_value=1.0): # Added clip_value
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.clip_value = clip_value # Store clip value

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Performs a forward pass through the network.
        X: Input data (batch_size, input_size)
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU activation

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y_true, y_pred):
        """
        Performs a backward pass (backpropagation) to update weights.
        X: Input data
        y_true: Target Q-values
        y_pred: Predicted Q-values
        """
        d_loss = y_pred - y_true 

        dW2 = np.dot(self.a1.T, d_loss)
        db2 = np.sum(d_loss, axis=0, keepdims=True)

        d_a1 = np.dot(d_loss, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0) # ReLU derivative

        dW1 = np.dot(X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # --- Gradient Clipping ---
        # Calculate the total norm of all gradients
        grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
        
        if grad_norm > self.clip_value:
            scale_factor = self.clip_value / grad_norm
            dW1 *= scale_factor
            db1 *= scale_factor
            dW2 *= scale_factor
            db2 *= scale_factor
        # --- End Gradient Clipping ---

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2


class MinesweeperAgent:
    """
    The Reinforcement Learning agent for Minesweeper.
    Uses a simple MLP to approximate Q-values.
    """
    def __init__(self, rows, cols, learning_rate=0.001, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.rows = rows
        self.cols = cols
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # print(f"DEBUG: MinesweeperAgent initialized with rows={self.rows}, cols={self.cols}") # Debug print - removed for less verbosity

        self.state_size = rows * cols * 3 
        self.action_size = rows * cols * 2 

        self.hidden_size = 256 
        self.model = MLP(self.state_size, self.hidden_size, self.action_size, self.learning_rate, clip_value=1.0) 

        self.replay_buffer = deque(maxlen=5000) 

    def _get_state_representation(self, game_board):
        """
        Converts the current game board state into a flat NumPy array suitable for the MLP.
        """
        state = np.zeros((self.rows, self.cols, 3))
        for r in range(self.rows):
            for c in range(self.cols):
                is_revealed = 1 if game_board.revealed[r][c] else 0
                is_flagged = 1 if game_board.flags[r][c] else 0
                
                cell_value = -2 
                if is_revealed:
                    cell_value = game_board.board[r][c] 
                    if cell_value == -1: 
                        cell_value = -1 
                
                state[r, c, 0] = is_revealed
                state[r, c, 1] = is_flagged
                state[r, c, 2] = cell_value
        
        return state.flatten().reshape(1, -1) 

    def choose_action(self, game_board):
        """
        Chooses an action using an epsilon-greedy policy.
        Returns: (row, col, action_type) where action_type is 0 for reveal, 1 for flag.
        """
        # print(f"DEBUG: choose_action called. Agent dimensions: {self.rows}x{self.cols}") # Debug print - removed for less verbosity
        valid_actions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if not game_board.revealed[r][c]: 
                    valid_actions.append((r, c, 0)) 
                    valid_actions.append((r, c, 1)) # Allow flagging/unflagging
        
        if not valid_actions:
            # print(f"DEBUG: choose_action: No valid actions found. All cells revealed or flagged. Game over: {game_board.game_over}") # Debug print - removed for less verbosity
            return None 
        
        # print(f"DEBUG: choose_action: Found {len(valid_actions)} valid actions.") # Debug print - removed for less verbosity

        if np.random.rand() < self.epsilon:
            action_idx = np.random.choice(len(valid_actions))
            chosen_action = valid_actions[action_idx]
            # print(f"DEBUG: choose_action (explore): Returning {chosen_action}") # Debug print - removed for less verbosity
            return chosen_action
        else:
            current_state = self._get_state_representation(game_board)
            q_values = self.model.forward(current_state)[0] 

            if np.isnan(q_values).any():
                print("DEBUG: WARNING: q_values contain NaN values in choose_action! Falling back to random action.")
                action_idx = np.random.choice(len(valid_actions))
                chosen_action = valid_actions[action_idx]
                # print(f"DEBUG: choose_action (NaN fallback): Returning random action {chosen_action}") # Debug print - removed for less verbosity
                return chosen_action

            best_action = None
            max_q_value = -float('inf')

            for r, c, action_type in valid_actions:
                action_idx = (r * self.cols + c) * 2 + action_type
                # Ensure we only consider actions that are actually valid and meaningful
                # If an action is to flag a cell, and that cell is already flagged, it's not a useful action.
                # However, the `flag_cell` method already handles this with a small penalty.
                # The primary issue is the agent getting stuck due to a high Q-value for a non-state-changing action.
                
                # We need to make sure that the action is actually "available" in the current state.
                # The `valid_actions` list already filters out revealed cells.
                # The `flag_cell` method gives a reward of 0 for toggling a flag, and -0.5 for unflagging.
                # If the agent keeps picking an action that results in 0 reward and no state change, it's stuck.
                # The penalty for unflagging should help, but if it keeps flagging the same unflagged cell,
                # it will still get 0 reward.
                
                # A more robust solution for "stuck" behavior often involves:
                # 1. More sophisticated state representation (e.g., historical actions).
                # 2. Reward shaping (e.g., small negative reward for *any* action that doesn't reveal a new cell).
                # 3. Exploration noise in exploitation (e.g., soft-max policy instead of greedy).
                # For now, let's rely on the penalty for unflagging and the overall game loss.

                if q_values[action_idx] > max_q_value:
                    max_q_value = q_values[action_idx]
                    best_action = (r, c, action_type)
            
            # print(f"DEBUG: choose_action (exploit): Returning {best_action}") # Debug print - removed for less verbosity
            return best_action
    
    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        """
        Learns from a batch of experiences sampled from the replay buffer.
        Performs a Q-learning update using the MLP.
        """
        if len(self.replay_buffer) < batch_size:
            return 

        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)

        current_q_values = self.model.forward(states)
        next_q_values = self.model.forward(next_states)
        
        max_next_q = np.max(next_q_values, axis=1, keepdims=True)
        target_q_values = rewards + self.discount_factor * max_next_q * (1 - dones)

        target_q_for_update = np.copy(current_q_values)
        
        for i, (r, c, action_type) in enumerate(actions):
            action_idx = (r * self.cols + c) * 2 + action_type
            target_q_for_update[i, action_idx] = target_q_values[i].item() 

        self.model.backward(states, target_q_for_update, current_q_values)

    def decay_epsilon(self):
        """Decays epsilon to reduce exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# --- Part 3: Graphical User Interface (GUI) ---

class MinesweeperGUI:
    """
    Handles the visual representation of the Minesweeper game using Tkinter.
    """
    def __init__(self, master, game_instance, cell_size=30):
        self.master = master
        self.game = game_instance
        self.cell_size = cell_size
        self.buttons = [] 

        self.master.title("Minesweeper RL")
        # Ensure geometry correctly calculates total width/height including padding
        self.master.geometry(f"{self.game.cols * self.cell_size}x{self.game.rows * self.cell_size + 50}") 

        self.frame = tk.Frame(master)
        # Removed self.frame.pack() - let grid manage the frame's position and size implicitly
        # Or, if you want to explicitly center the frame, use grid on the master itself
        self.frame.grid(row=0, column=0, sticky="nsew") # Use grid for the frame in the master window
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.status_label = tk.Label(master, text="Playing...", font=("Arial", 14))
        self.status_label.grid(row=1, column=0, columnspan=self.game.cols, sticky="ew", pady=5) # Place status label below grid

        self._create_widgets()

    def _handle_click(self, r, c):
        """Handles a manual click on a cell (for testing/debugging)."""
        if self.game.game_over:
            return
        
        reward, game_ended = self.game.reveal_cell(r, c)
        self.update_board_display()
        self._update_status(game_ended)

    def update_board_display(self, highlight_cell=None):
        """
        Updates the visual state of the board based on the game's internal state.
        Optionally highlights a specific cell.
        """
        for r in range(self.game.rows):
            for c in range(self.game.cols): 
                button = self.buttons[r][c]
                button.config(relief=tk.RAISED, bg="lightgray", fg="black", text="") 

                if self.game.revealed[r][c]:
                    button.config(relief=tk.SUNKEN, bg="white")
                    if self.game.board[r][c] == -1:
                        button.config(text="*", fg="red", font=("Arial", 12, "bold")) 
                    elif self.game.board[r][c] > 0:
                        button.config(text=str(self.game.board[r][c]), font=("Arial", 12, "bold"))
                        colors = {1: "blue", 2: "green", 3: "red", 4: "purple", 5: "maroon", 6: "turquoise", 7: "black", 8: "gray"}
                        button.config(fg=colors.get(self.game.board[r][c], "black"))
                elif self.game.flags[r][c]:
                    button.config(text="F", fg="orange", font=("Arial", 12, "bold")) 
                
                if highlight_cell and highlight_cell == (r, c):
                    button.config(bg="yellow") 

        if self.game.game_over and not self.game.win:
            for r in range(self.game.rows):
                for c in range(self.game.cols):
                    if self.game.board[r][c] == -1 and not self.game.revealed[r][c]:
                        self.buttons[r][c].config(text="*", fg="red", bg="lightcoral", font=("Arial", 12, "bold"))

        self.master.update_idletasks() 

    def _update_status(self, game_ended):
        """Updates the status label at the bottom of the GUI."""
        if game_ended:
            if self.game.win:
                self.status_label.config(text="You Win!", fg="green")
            else:
                self.status_label.config(text="Game Over!", fg="red")
        else:
            self.status_label.config(text="Playing...", fg="black")

    def _create_widgets(self):
        """Creates the grid of buttons for the Minesweeper board."""
        for row_buttons in self.buttons:
            for button in row_buttons:
                button.destroy()
        self.buttons = []

        for r in range(self.game.rows):
            row_buttons = []
            for c in range(self.game.cols):
                button = tk.Button(self.frame, width=4, height=2,
                                   command=lambda r=r, c=c: self._handle_click(r, c))
                button.grid(row=r, column=c, padx=1, pady=1) # Added padx/pady for spacing
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_display()


# --- Part 4: Training Loop and Visualization Integration ---

def run_rl_training(rows, cols, num_mines, num_episodes, max_steps_per_episode,
                    learning_rate, discount_factor, epsilon_start, epsilon_decay, min_epsilon,
                    batch_size, gui_delay_ms=50):
    """
    Main function to run the RL training loop and visualize the agent's actions.
    """
    root = tk.Tk()
    root.withdraw() 

    agent = MinesweeperAgent(rows, cols, learning_rate, discount_factor, 
                             epsilon_start, epsilon_decay, min_epsilon)
    
    game_window = tk.Toplevel(root)
    game_window.protocol("WM_DELETE_WINDOW", root.destroy) 

    win_rates = []
    total_wins = 0

    print("Starting RL training...")
    for episode in range(num_episodes):
        game = MinesweeperGame(rows, cols, num_mines) 
        if 'gui' in locals() and gui.master:
            gui.master.destroy() 
        
        gui_master = tk.Toplevel(root)
        gui_master.protocol("WM_DELETE_WINDOW", root.destroy) 
        gui = MinesweeperGUI(gui_master, game)

        print(f"--- Episode {episode + 1} Reset ---")
        print(f"Initial revealed count: {sum(sum(row) for row in game.revealed)}")
        print(f"Initial flagged count: {sum(sum(row) for row in game.flags)}")
        print(f"Initial game_over status: {game.game_over}")

        current_state = agent._get_state_representation(game)
        episode_reward = 0
        steps_taken = 0

        gui.update_board_display() 
        gui.status_label.config(text=f"Episode {episode + 1}/{num_episodes} (Epsilon: {agent.epsilon:.2f})")
        gui.master.update_idletasks()
        time.sleep(0.5) 

        for step in range(max_steps_per_episode):
            if game.game_over:
                print(f"DEBUG: Game over before step {step+1} in episode {episode+1}. Win: {game.win}")
                break

            action = agent.choose_action(game)
            # print(f"DEBUG: After choose_action, 'action' is: {action}") # Removed for less verbosity

            if action is None: 
                print(f"DEBUG: Action is None (no valid actions) before step {step+1} in episode {episode+1}. Game over: {game.game_over}")
                # More detailed debug for why action might be None
                if not any(not cell for row in game.revealed for cell in row):
                    print("DEBUG: All cells are revealed, so no valid actions.")
                else:
                    print("DEBUG: Valid actions were found by choose_action, but it still returned None. This indicates NaN issue or similar.")

                break
            
            r, c, action_type = action
            
            reward = 0
            game_ended = False
            if action_type == 0: # Reveal
                reward, game_ended = game.reveal_cell(r, c)
            elif action_type == 1: # Flag/Unflag
                reward, game_ended = game.flag_cell(r, c)
            
            episode_reward += reward
            steps_taken += 1

            next_state = agent._get_state_representation(game)
            agent.remember(current_state, action, reward, next_state, game_ended)
            current_state = next_state

            gui.update_board_display(highlight_cell=(r, c))
            gui.status_label.config(text=f"Episode {episode + 1}/{num_episodes} | Step {step + 1} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
            gui.master.update_idletasks()
            time.sleep(gui_delay_ms / 1000.0) 

            if len(agent.replay_buffer) > batch_size: 
                agent.learn(batch_size)
        
        if game.win:
            total_wins += 1
            print(f"Episode {episode + 1}: WIN! Total Reward: {episode_reward:.2f}, Steps: {steps_taken}")
            gui.status_label.config(text=f"Episode {episode + 1}: WIN! (Total Wins: {total_wins})", fg="green")
        else:
            print(f"Episode {episode + 1}: LOSE. Total Reward: {episode_reward:.2f}, Steps: {steps_taken}")
            gui.status_label.config(text=f"Episode {episode + 1}: LOSE. (Total Wins: {total_wins})", fg="red")
        
        gui.master.update_idletasks()
        time.sleep(1.0) 

        agent.decay_epsilon() 

        if (episode + 1) % 10 == 0: 
            win_rate = total_wins / (episode + 1)
            win_rates.append(win_rate)
            print(f"Current Win Rate (last {episode+1} episodes): {win_rate:.2f}")

    print("\nTraining finished.")
    print(f"Final Win Rate: {total_wins / num_episodes:.2f} ({total_wins}/{num_episodes} wins)")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot([i * 10 for i in range(len(win_rates))], win_rates)
        plt.title("RL Agent Win Rate Over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Win Rate")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not found. Cannot plot win rate.")

    root.mainloop() 


if __name__ == "__main__":
    BOARD_ROWS = 8
    BOARD_COLS = 8
    NUM_MINES = 10

    NUM_EPISODES = 5000 
    MAX_STEPS_PER_EPISODE = BOARD_ROWS * BOARD_COLS * 3 # Increased max steps slightly
    # Note: If the agent still gets stuck, consider increasing this further or implementing
    # a "stuck" detection mechanism to force a random action after N repeated actions.

    LEARNING_RATE = 0.00001 # Retained very low learning rate
    DISCOUNT_FACTOR = 0.95 
    EPSILON_START = 1.0  
    EPSILON_DECAY = 0.995 
    MIN_EPSILON = 0.01   
    BATCH_SIZE = 64 
    GUI_DELAY_MS = 100 

    run_rl_training(BOARD_ROWS, BOARD_COLS, NUM_MINES, NUM_EPISODES, MAX_STEPS_PER_EPISODE,
                    LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_DECAY, MIN_EPSILON,
                    BATCH_SIZE, GUI_DELAY_MS)

