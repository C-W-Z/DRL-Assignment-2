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
import gc
import atexit

gc.collect()

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

@njit
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

@njit
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

@njit
def _add_random_tile(board):
    # Placeholder: add a random tile (2 or 4) to an empty cell
    empty = np.where(board == 0)
    if len(empty[0]) > 0:
        idx = np.random.choice(len(empty[0]))
        x, y = empty[0][idx], empty[1][idx]
        board[x, y] = 2 if np.random.random() < 0.9 else 4
    return board

@njit
def _is_move_legal(board, action, score):
    _, moved, _ = move_board(board.copy(), action, score)
    return moved

@njit
def get_legal_moves(board, score):
    return [a for a in range(4) if _is_move_legal(board, a, score)]

@njit
def _is_game_over(board):
        if np.any(board == 0):
            return False
        size = board.shape[0]
        for i in range(size):
            for j in range(size - 1):
                if board[i, j] == board[i, j+1]:
                    return False
        for j in range(size):
            for i in range(size - 1):
                if board[i, j] == board[i+1, j]:
                    return False
        return True

@njit
def _step(board, action, score, afterstate=False):
        board, moved, score = move_board(board, action, score)

        if moved and not afterstate:
            _add_random_tile(board)

        done = _is_game_over(board)

        return board, score, done, moved

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
        self.board = _add_random_tile(self.board)

    def is_game_over(self):
        return _is_game_over(self.board)

    def step(self, action, afterstate=False):
        self.board, self.score, done, self.last_move_valid = _step(self.board, action, self.score, afterstate)
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

    def is_move_legal(self, action):
        return _is_move_legal(self.board, action, self.score)

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
        self.board = _add_random_tile(self.board)

    def is_game_over(self):
        return _is_game_over(self.board)

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
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
    [(0,0), (0,1), (1,0), (1,1), (0,2), (0,3)],
    [(1,0), (1,1), (0,1), (2,1), (1,2), (1,3)],
    [(0,0), (1,1), (2,2), (0,1), (1,2), (2,3)],
]

# patterns = [
#     [(0,0), (0,1)],
#     [(1,0), (1,1)],
#     [(0,0), (1,1), (2,2)],
#     [(0,0), (0,1), (0,2), (0,3)],
#     [(1,0), (1,1), (1,2), (1,3)],
#     [(1,0), (0,0), (0,1), (0,2)],
#     [(2,1), (1,1), (1,2), (1,3)],
#     [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
#     [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
# ]

approximator = None

def init_model():
    global approximator
    if approximator is None:
        gc.collect()
        approximator = NTupleApproximator(board_size=4, patterns=patterns)
        with open("Q1_2048_approximator_weights_40000.pkl", "rb") as f:
            approximator.weights = pickle.load(f)

@njit
def calculate_ucb1(total_reward, visits, parent_visits):
    if visits == 0:
        return float('inf')
    return (total_reward / visits) + np.sqrt(2.0 * np.log(parent_visits) / visits)

@njit
def calculate_untried_tiles(board, max_samples=10):
    empty_cells = np.where(board == 0)
    x_coords, y_coords = empty_cells[0], empty_cells[1]
    n_empty = len(x_coords)
    untried = []

    if n_empty <= max_samples:
        for i in range(n_empty):
            value = 2 if np.random.random() < 0.9 else 4
            untried.append((x_coords[i], y_coords[i], value))
    else:
        ids = np.random.choice(n_empty, max_samples, replace=False)
        for i in ids:
            value = 2 if np.random.random() < 0.9 else 4
            untried.append((x_coords[i], y_coords[i], value))

    return untried

class AfterstateNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.env = Game2048EnvNoRandom()
        self.env.board = state
        self.env.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_random_tiles = []

    def calculate_untried_random_tiles(self):
        self.untried_random_tiles = calculate_untried_tiles(self.env.board)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_random_tiles) == 0

class ChanceNode:
    def __init__(self, state, score, parent=None, child_id=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        child_id: index of this node in the parent's children list
        """
        self.env = Game2048EnvNoRandom()
        self.env.board = state
        self.env.score = score
        self.parent = parent
        self.child_id = child_id
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = get_legal_moves(state, score)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, approximator, iterations=500, rollout_depth=10):
        self.approximator = approximator
        self.iterations = iterations
        # self.c = exploration_constant
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = Game2048EnvNoRandom()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def get_best_child(self, node):
        # Select the child node with the highest UCB1 value.
        if node.visits == 0:
            return random.choice(list(node.children.values()))
        return max(node.children.values(), key=lambda child: calculate_ucb1(child.total_reward, child.visits, node.visits))

    def expand(self, node):
        if node.env.is_game_over():
            return node
        if isinstance(node, AfterstateNode):
            if node.untried_random_tiles:
                tile = random.choice(node.untried_random_tiles)
                node.untried_random_tiles.remove(tile)
                x, y, value = tile
                new_node = ChanceNode(copy.deepcopy(node.env.board), node.env.score, node, tile)
                new_node.env.board[x, y] = value
                node.children[tile] = new_node
                node = new_node
        elif isinstance(node, ChanceNode):
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_node = AfterstateNode(copy.deepcopy(node.env.board), node.env.score, node, action)
                new_node.env.step(action)
                new_node.calculate_untried_random_tiles()
                node.children[action] = new_node
                node = new_node
                if not node.untried_random_tiles and not node.children:
                    new_node = ChanceNode(copy.deepcopy(node.env.board), node.env.score, node, 0)
                    node.children[0] = new_node
                    node = new_node
        return node

    def traverse(self, node):
        while True:
            if node.env.is_game_over():
                return node
            if not node.fully_expanded():
                return self.expand(node)
            node = self.get_best_child(node)

    def rollout(self, afterstate: bool, sim_env: Game2048EnvNoRandom, depth):
        new_score = 0
        if afterstate and not sim_env.is_game_over():
            sim_env.add_random_tile()
        for _ in range(depth):
            legal_actions = get_legal_moves(sim_env.board, sim_env.score)
            if not legal_actions:
                break
            best_value = -np.inf
            best_env = None
            for a in legal_actions:
                temp_env = self.create_env_from_state(sim_env.board, sim_env.score)
                next_state, _, _, _ = temp_env.step(a)
                value = approximator.value(next_state)
                if value > best_value:
                    best_value = value
                    best_env = temp_env
            sim_env = best_env
            sim_env.add_random_tile()
            if sim_env.is_game_over():
                break
        return new_score + self.approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def run_simulation(self, root):
        node = root
        node = self.traverse(node)
        sim_env = self.create_env_from_state(node.env.board, node.env.score)
        rollout_reward = self.rollout(isinstance(node, AfterstateNode), sim_env, self.rollout_depth)
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

def get_action(state, score):
    init_model()

    # env = Game2048Env()
    # env.board = state
    # env.score = score
    # legal_moves = [a for a in range(4) if env.is_move_legal(a)]

    # values = []
    # for a in legal_moves:
    #     temp_env = Game2048EnvNoRandom()
    #     temp_env.board = state
    #     temp_env.score = score
    #     next_state, _, _, _ = temp_env.step(a)
    #     values.append(approximator.value(next_state))
    # best_action = legal_moves[np.argmax(values)]
    # return best_action

    td_mcts = TD_MCTS(approximator, iterations=100, rollout_depth=5)

    root = ChanceNode(state, score)
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
