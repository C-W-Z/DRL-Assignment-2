# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict
from numba import jit, njit

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

@jit(nopython=True)
def compress_and_merge(row, score):
    size = 4
    # 過濾非零元素
    temp = np.zeros(size, dtype=np.int32)
    pos = 0
    for i in range(size):
        if row[i] != 0:
            temp[pos] = row[i]
            pos += 1

    # 合併相鄰相同元素
    result = np.zeros(size, dtype=np.int32)
    write_pos = 0
    i = 0
    while i < pos:
        if i + 1 < pos and temp[i] == temp[i + 1]:
            result[write_pos] = temp[i] * 2
            score += temp[i] * 2
            i += 2
        else:
            result[write_pos] = temp[i]
            i += 1
        write_pos += 1

    return result, score

@jit(nopython=True)
def move_board(board, direction, score):
    new_board = board.copy()
    moved = False
    if direction == 0:  # 上
        for j in range(4):
            col, new_score = compress_and_merge(new_board[:, j], score)
            if not np.array_equal(col, new_board[:, j]):
                moved = True
            new_board[:, j] = col
            score = new_score
    elif direction == 1:  # 下
        for j in range(4):
            col = new_board[::-1, j]
            col, new_score = compress_and_merge(col, score)
            if not np.array_equal(col, new_board[::-1, j]):
                moved = True
            new_board[::-1, j] = col
            score = new_score
    elif direction == 2:  # 左
        for i in range(4):
            row, new_score = compress_and_merge(new_board[i], score)
            if not np.array_equal(row, new_board[i]):
                moved = True
            new_board[i] = row
            score = new_score
    elif direction == 3:  # 右
        for i in range(4):
            row = new_board[i, ::-1]
            row, new_score = compress_and_merge(row, score)
            if not np.array_equal(row, new_board[i, ::-1]):
                moved = True
            new_board[i, ::-1] = row
            score = new_score
    return new_board, moved, score

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = np.where(self.board == 0)
        if len(empty_cells[0]) > 0:
            idx = random.randint(0, len(empty_cells[0]) - 1)
            x, y = empty_cells[0][idx], empty_cells[1][idx]
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.board, moved, self.score = move_board(self.board, action, self.score)

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row


    def is_move_legal(self, action):
        temp_board = self.board.copy()
        new_board, moved, _ = move_board(temp_board, action, self.score)
        return moved

class Game2048EnvNoRandom(gym.Env):
    def __init__(self):
        super(Game2048EnvNoRandom, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = np.where(self.board == 0)
        if len(empty_cells[0]) > 0:
            idx = random.randint(0, len(empty_cells[0]) - 1)
            x, y = empty_cells[0][idx], empty_cells[1][idx]
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.board, moved, self.score = move_board(self.board, action, self.score)

        self.last_move_valid = moved

        # if moved:
        #     self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()
        new_board, moved, _ = move_board(temp_board, action, self.score)
        return moved

import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rot90(pattern, board_size):
    return [(j, board_size - 1 - i) for (i, j) in pattern]

def rot180(pattern, board_size):
    return [(board_size - 1 - i, board_size - 1 - j) for (i, j) in pattern]

def rot270(pattern, board_size):
    return [(board_size - 1 - j, i) for (i, j) in pattern]

def reflect_h(pattern, board_size):
    return [(i, board_size - 1 - j) for (i, j) in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.symmetry_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups.append(syms)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

        self.tile_to_index_lookup = {0: 0}
        max_tile = 2 ** 20
        for i in range(1, 21):
            self.tile_to_index_lookup[2 ** i] = i

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        board_size = self.board_size
        r90 = rot90(pattern, board_size)
        r180 = rot180(pattern, board_size)
        r270 = rot270(pattern, board_size)
        return [
            pattern,
            r90,
            r180,
            r270,
            reflect_h(pattern, board_size),
            reflect_h(r90, board_size),
            reflect_h(r180, board_size),
            reflect_h(r270, board_size)
        ]

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        # if tile == 0:
        #     return 0
        # else:
        #     return int(math.log(tile, 2))
        return self.tile_to_index_lookup[tile]

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[i, j]) for (i, j) in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0.0
        for i, syms in enumerate(self.symmetry_groups):
            group_value = 0.0
            for pattern in syms:
                feature = self.get_feature(board, pattern)
                group_value += self.weights[i][feature]
            total_value += group_value / len(syms)
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, syms in enumerate(self.symmetry_groups):
            update_value = alpha * delta / len(syms)
            for pattern in syms:
                feature = self.get_feature(board, pattern)
                self.weights[i][feature] += update_value

def td_learning(env, approximator, previous_episodes=0, num_episodes=50000, alpha=0.01, gamma=0.99):
    """
    Trains the 2048 agent using TD-Learning with afterstate updates.
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        previous_score = 0
        done = False
        max_tile = np.max(state)
        # previous_afterstate = state
        # trajectory = []

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # Collect afterstates and their values
            afterstates = []
            afterstate_values = []

            for a in legal_moves:
                env_copy = Game2048EnvNoRandom()
                env_copy.board = env.board.copy()
                env_copy.score = env.score
                afterstate, _, _, _ = env_copy.step(a)
                afterstates.append((afterstate, a))
                afterstate_values.append(approximator.value(afterstate))

            idx = np.argmax(afterstate_values)
            selected_afterstate, action = afterstates[idx]
            selected_value = afterstate_values[idx]

            # Take the action in the real environment
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # trajectory.append((previous_afterstate, action, incremental_reward, selected_afterstate, done))

            # previous_afterstate = selected_afterstate

            # Update the value function for the selected afterstate
            # The target is the immediate reward plus discounted value of next afterstate
            if done:
                target = incremental_reward
            else:
                # For the next state, we need to look at possible future afterstates
                next_values = []
                next_legal_moves = [a for a in range(4) if env.is_move_legal(a)]

                if next_legal_moves:
                    for a in next_legal_moves:
                        env_copy = Game2048EnvNoRandom()
                        env_copy.board = env.board.copy()
                        env_copy.score = env.score
                        future_state, _, _, _ = env_copy.step(a)
                        next_values.append(approximator.value(future_state))

                    future_value = max(next_values) if next_values else 0
                    target = incremental_reward + gamma * future_value
                else:
                    target = incremental_reward

            # Update the value function
            delta = target - selected_value
            approximator.update(selected_afterstate, delta, alpha)

            state = next_state

        # for previous_afterstate, action, incremental_reward, afterstate, done in reversed(trajectory):
        #     current_value = approximator.value(previous_afterstate)
        #     next_value = approximator.value(afterstate) if not done else 0
        #     td_error = incremental_reward + gamma * next_value - current_value
        #     approximator.update(previous_afterstate, td_error, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {previous_episodes+episode+1}/{previous_episodes+num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}", flush=True)

        if (episode + 1) % 1000 == 0:
            with open(f"Q1_2048_approximator_weights_{previous_episodes+episode+1}.pkl", "wb") as f:
                pickle.dump(approximator.weights, f)

    return final_scores

# TODO: Define your own n-tuple patterns
patterns = [
    # [(0,0)],
    # [(0,1)],
    # [(1,0)],
    # [(1,1)],
    [(0,0), (0,1)],
    [(1,0), (1,1)],
    [(0,0), (1,1), (2,2)],
    [(0,0), (0,1), (0,2), (0,3)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(1,0), (0,0), (0,1), (0,2)],
    [(2,1), (1,1), (1,2), (1,3)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

with open("Q1_2048_approximator_weights_4000.pkl", "rb") as f:
    approximator.weights = pickle.load(f)

# UCT Node for MCTS
class UCTNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        env = Game2048Env()
        env.board = state
        env.score = score
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        env = Game2048Env()
        env.board = state
        env.score = score
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = Game2048EnvNoRandom()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.visits == 0:
          return random.choice(list(node.children.values()))
        log_N = math.log(node.visits)
        return max(node.children.values(), key=lambda child: (child.total_reward / child.visits) + self.c * math.sqrt(log_N / child.visits))

    def rollout(self, sim_env: Game2048EnvNoRandom, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        total_reward = 0
        discount = 1.0
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            board, _, _, _ = sim_env.step(action)
            total_reward += self.approximator.value(board) * discount
            discount *= self.gamma
            sim_env.add_random_tile()
            if sim_env.is_game_over():
                break
        return total_reward + discount * self.approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def run_simulation(self, root: TD_MCTS_Node):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)
            sim_env.add_random_tile()

        # TODO: Expansion: if the node has untried actions, expand an untried action.
        if not sim_env.is_game_over() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, new_score, _, _ = sim_env.step(action)
            sim_env.add_random_tile()
            new_node = UCTNode(new_state, new_score, node, action)
            node.children[action] = new_node
            node = new_node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

td_mcts = TD_MCTS(approximator, iterations=50, exploration_constant=math.sqrt(2), rollout_depth=10, gamma=0.99)

def get_action(state, score):

    root = TD_MCTS_Node(state, score)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act

if __name__ == "__main__":
    # env = Game2048Env()
    # state = env.reset()
    # # env.render()

    # done = False
    # while not done:
    #     best_act = get_action(state, env.score)
    #     print(f"TD-MCTS selected action: {best_act}, Score: {env.score}")

    #     # Execute the selected action and update the state
    #     state, reward, done, _ = env.step(best_act)
    #     if done:
    #         env.render(action=best_act)

    # print("Game over, final score:", env.score)

    final_scores = td_learning(Game2048Env(), approximator, previous_episodes=1000, num_episodes=2000, alpha=0.1, gamma=0.99)

    avg_scores = []
    for i in range(0, len(final_scores), 100):
        avg_score = np.mean(final_scores[i:i+100])
        avg_scores.append(avg_score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(final_scores)), final_scores, label='Score')
    plt.plot(range(100, len(final_scores) + 1, 100), avg_scores, label='Average Score (per 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score Over Training Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
