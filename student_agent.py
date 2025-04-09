import numpy as np
from Game2048Env import Game2048Env, get_legal_moves
from TDL2048 import board, learning, pattern, move
import gc, atexit, random
from numba import njit

def numpy_to_bitboard(np_board):
    b = board()
    for i in range(4):
        for j in range(4):
            value = np_board[i, j]
            if value != 0:
                tile = int(np.log2(value))
                b.set(i * 4 + j, tile)
    return b

@njit
def tdl_action_to_gym_action(tdl_action):
    mapping = {0: 0, 1: 3, 2: 1, 3: 2}
    return mapping[tdl_action]

tdl = None

def init_model():
    global tdl
    if tdl is None:
        gc.collect()
        tdl = learning()
        board.lookup.init()

        tdl.add_feature(pattern([0, 1, 2, 3, 4, 5]))
        tdl.add_feature(pattern([4, 5, 6, 7, 8, 9]))
        tdl.add_feature(pattern([0, 1, 2, 4, 5, 6]))
        tdl.add_feature(pattern([4, 5, 6, 8, 9, 10]))

        tdl.load("2048.bin")
        gc.collect()

def cleanup():
    global tdl
    del tdl
    gc.collect()

atexit.register(cleanup)

def get_tdl_action(state):
    bitboard_state = numpy_to_bitboard(state)
    best_move = tdl.select_best_move(bitboard_state)
    tdl_action = best_move.action()
    return tdl_action_to_gym_action(tdl_action)

def tdl_estimate(state):
    b = numpy_to_bitboard(state)
    mv = move(b)
    return tdl.estimate(mv.afterstate())

@njit
def calculate_ucb1(total_reward, visits, parent_visits, coef=1.41):
    if visits == 0:
        return np.inf
    return (total_reward / visits) + coef * np.sqrt(np.log(parent_visits) / visits)
    # return (total_reward / visits) + (mean_reward) * np.sqrt(np.log(parent_visits) / visits / 2.0)

@njit
def calculate_untried_tiles(board, max_samples=10):
    empty_cells = np.where(board == 0)
    x_coords, y_coords = empty_cells[0], empty_cells[1]
    n_empty = len(x_coords)
    untried = []

    if n_empty <= max_samples:
        for i in range(n_empty):
            untried.append((x_coords[i], y_coords[i], 2))
            untried.append((x_coords[i], y_coords[i], 4))
    else:
        ids = np.random.choice(n_empty, max_samples, replace=False)
        for i in ids:
            untried.append((x_coords[i], y_coords[i], 2))
            untried.append((x_coords[i], y_coords[i], 4))

    return untried

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
        # List of untried actions based on the current state's legal moves
        self.untried_actions = get_legal_moves(state, score)

    def calculate_untried_actions(self, state, score):
        self.state = state
        self.score = score
        # List of untried actions based on the current state's legal moves
        self.untried_actions = get_legal_moves(state, score)
        for c in self.children.values():
            if c.action in self.untried_actions:
                self.untried_actions.remove(c.action)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99, coef=1.41):
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.coef = coef

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = Game2048Env()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.visits == 0:
            return random.choice(list(node.children.values()))
        # mean = sum(child.total_reward / child.visits for child in node.children.values()) / len(node.children)
        mean = min(child.total_reward / child.visits for child in node.children.values())
        return max(node.children.values(), key=lambda child: calculate_ucb1(child.total_reward, child.visits, node.visits, self.coef))

    def rollout(self, sim_env: Game2048Env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        reward = tdl_estimate(sim_env.board)
        for _ in range(depth):
            legal_actions = get_legal_moves(sim_env.board, sim_env.score)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            # action = get_tdl_action(sim_env.board)
            _, _, _, _ = sim_env.step(action, True)
            reward = tdl_estimate(sim_env.board)
            sim_env.add_random_tile()
            if sim_env.is_game_over():
                break
        return sim_env.score + reward

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

        node.calculate_untried_actions(sim_env.board, sim_env.score)

        # TODO: Expansion: if the node has untried actions, expand an untried action.
        if not sim_env.is_game_over() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, new_score, _, _ = sim_env.step(action)
            new_node = TD_MCTS_Node(new_state, new_score, node, action)
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

def get_action(state, score):
    init_model()
    td_mcts = TD_MCTS(iterations=2000, rollout_depth=0, coef=np.sqrt(2) * 10000)

    root = TD_MCTS_Node(state, score)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_act, _ = td_mcts.best_action_distribution(root)

    print([c.visits for c in root.children.values()], score, root.total_reward / root.visits, [c.total_reward / c.visits for c in root.children.values()])

    return best_act

if __name__ == "__main__":
    env = Game2048Env()
    env.reset()
    done = False
    while not done:
        if not env.legal_moves():
            break
        action = get_action(env.board)
        _, _, done, _ = env.step(action)
    print(f"Game Over! Final Score: {env.score}")
