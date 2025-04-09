import numpy as np
from Game2048Env import Game2048Env, get_legal_moves, filter_moves_keep_max_in_corner
from TDL2048 import board, learning, pattern, move
import gc, random
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

        # tdl.add_feature(pattern([0, 1, 2, 3, 4, 5]))
        # tdl.add_feature(pattern([4, 5, 6, 7, 8, 9]))
        # tdl.add_feature(pattern([0, 1, 2, 4, 5, 6]))
        # tdl.add_feature(pattern([4, 5, 6, 8, 9, 10]))

        tdl.add_feature(pattern([ 0, 1, 2, 4, 5, 6 ]))
        tdl.add_feature(pattern([ 0, 1, 2, 3, 4, 5 ]))
        tdl.add_feature(pattern([ 4, 5, 6, 7, 8, 9 ]))
        tdl.add_feature(pattern([ 0, 1, 5, 6, 7, 10 ]))
        tdl.add_feature(pattern([ 0, 1, 2, 5, 9, 10 ]))
        tdl.add_feature(pattern([ 0, 1, 5, 9, 13, 14 ]))
        tdl.add_feature(pattern([ 0, 1, 5, 8, 9, 13 ]))
        tdl.add_feature(pattern([ 0, 1, 2, 4, 6, 10 ]))

        tdl.load("2048_8x6.bin")

def get_tdl_action(state):
    bitboard_state = numpy_to_bitboard(state)
    best_move = tdl.select_best_move(bitboard_state)
    tdl_action = best_move.action()
    return tdl_action_to_gym_action(tdl_action)

def tdl_estimate(state):
    b = numpy_to_bitboard(state)
    mv = move(b)
    return tdl.estimate(mv.afterstate())

def filter_moves_by_td_threshold(board, score, legal_moves, threshold_rate=0.8):
    if not legal_moves:
        return []

    td_values = []
    for action in legal_moves:
        temp_env = Game2048Env()
        temp_env.board = board.copy()
        temp_env.score = score
        temp_env.step(action, True)
        td_value = temp_env.score + tdl_estimate(temp_env.board)
        td_values.append((action, td_value))

    max_td = max(v for _, v in td_values)
    threshold = max_td * threshold_rate

    valid_moves = [action for action, td_value in td_values if td_value >= threshold]
    return valid_moves

@njit
def calculate_ucb1(total_reward, visits, parent_visits, constant):
    if visits == 0:
        return np.inf
    # return (total_reward / visits - score) + (mean_reward - score) * np.sqrt(np.log(parent_visits) / visits / 2.0)
    return np.log10(total_reward / visits) + constant * np.sqrt(np.log(parent_visits) / visits)

@njit
def calculate_untried_tiles(board, max_samples=32):
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

class AfterstateNode:
    def __init__(self, state, score, parent=None, action=None, max_samples=10):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.env = Game2048Env()
        self.env.board = state.copy()
        self.env.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_random_tiles = []
        self.max_samples = max_samples

    def calculate_untried_random_tiles(self):
        self.untried_random_tiles = calculate_untried_tiles(self.env.board, self.max_samples)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_random_tiles) == 0

class ChanceNode:
    def __init__(self, state, score, parent=None, tile=2):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        child_id: index of this node in the parent's children list
        """
        self.env = Game2048Env()
        self.env.board = state.copy()
        self.env.score = score
        self.parent = parent
        # self.child_id = child_id
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = []
        self.possibility = 0.9 if tile == 2 else 0.1
        if tile == 0:
            self.possibility = 1.0

    def calculate_untried_actions(self, threshold_rate=0.8):
        self.untried_actions = self.env.legal_moves()
        self.untried_actions = filter_moves_by_td_threshold(self.env.board, self.env.score, self.untried_actions, threshold_rate)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, iterations=500, rollout_depth=10, max_samples=32, constant=1.41, threshold_rate=0.8):
        self.iterations = iterations
        # self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.max_samples = max_samples
        self.constant = constant
        self.threshold_rate = threshold_rate

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = Game2048Env()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def get_best_child(self, node):
        # Select the child node with the highest UCB1 value.
        if node.visits == 0:
            return random.choice(list(node.children.values()))
        if isinstance(node, AfterstateNode):
            child_list = list(node.children.values())
            poss = np.array([c.possibility for c in child_list], dtype=np.float32)
            poss /= np.sum(poss)
            return np.random.choice(child_list, p=poss)
        return max(node.children.values(), key=lambda child: calculate_ucb1(child.total_reward, child.visits, node.visits,  self.constant))

    def expand(self, node):
        if node.env.is_game_over():
            return node

        if isinstance(node, ChanceNode):
            # assert(len(node.children) + len(node.untried_actions) == len(get_legal_moves(node.env.board, node.env.score)))
            if node.untried_actions:
                values = []
                for a in node.untried_actions:
                    temp_env = self.create_env_from_state(node.env.board, node.env.score)
                    temp_env.step(a, True)
                    values.append(temp_env.score + tdl_estimate(temp_env.board))
                action = node.untried_actions[np.argmax(values)]
                # action = random.choice(node.untried_actions)
                # action = node.untried_actions[0]
                node.untried_actions.remove(action)
                new_node = AfterstateNode(node.env.board, node.env.score, node, action, self.max_samples)
                new_node.env.step(action, True)
                new_node.calculate_untried_random_tiles()
                node.children[action] = new_node
                node = new_node
                if not node.untried_random_tiles and not node.children:
                    new_node = ChanceNode(node.env.board, node.env.score, node, 0)
                    node.children[0] = new_node
                    new_node.calculate_untried_actions(self.threshold_rate)
                    node = new_node

        if isinstance(node, AfterstateNode):
            # return node
            if node.untried_random_tiles:
                # 展開所有節點
                # for tile in node.untried_random_tiles:
                #     x, y, value = tile
                #     new_node = ChanceNode(node.env.board, node.env.score, node, value)
                #     new_node.env.board[x, y] = value
                #     node.children[tile] = new_node
                #     new_node.calculate_untried_actions(self.threshold_rate)
                    # sim_env = self.create_env_from_state(new_node.env.board, new_node.env.score)
                    # rollout_reward = self.rollout(False, sim_env, self.rollout_depth)
                    # constant = 0.9 if value == 2 else 0.1
                    # self.backpropagate(new_node, rollout_reward * constant, constant)
                # node.untried_random_tiles = []
                # return self.get_best_child(node)
                poss = np.array([t[2] for t in node.untried_random_tiles], dtype=np.float32)
                poss /= np.sum(poss)
                tileid = np.random.choice(len(node.untried_random_tiles), p=poss)
                tile = node.untried_random_tiles[tileid]
                node.untried_random_tiles.remove(tile)
                x, y, value = tile
                new_node = ChanceNode(node.env.board, node.env.score, node, value)
                new_node.env.board[x, y] = value
                node.children[tile] = new_node
                new_node.calculate_untried_actions(self.threshold_rate)
                node = new_node

        return node

    def traverse(self, node):
        while True:
            if node.env.is_game_over():
                return node
            if not node.fully_expanded():
                return self.expand(node)
            node = self.get_best_child(node)

    def rollout(self, is_afterstate: bool, sim_env: Game2048Env, depth):
        estim = tdl_estimate(sim_env.board)
        if is_afterstate and not sim_env.is_game_over():
            sim_env.add_random_tile()
        for _ in range(depth):
            legal_actions = sim_env.legal_moves()
            if not legal_actions:
                break
            valid_actions = filter_moves_by_td_threshold(sim_env.board, sim_env.score, legal_actions, self.threshold_rate)
            a = random.choice(valid_actions)
            # a = get_tdl_action(sim_env.board)
            sim_env.step(a, True)
            estim = tdl_estimate(sim_env.board)
            sim_env.add_random_tile()
            if sim_env.is_game_over():
                break
        return sim_env.score + estim

    def backpropagate(self, node, reward, count=1):
        current = node
        while current is not None:
            current.visits += count
            current.total_reward += reward
            current = current.parent

    def run_simulation(self, root):
        node = root
        node = self.traverse(node)
        sim_env = self.create_env_from_state(node.env.board, node.env.score)
        rollout_reward = self.rollout(isinstance(node, AfterstateNode), sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        return max(root.children.items(), key=lambda child: child[1].total_reward / child[1].visits)

root = None
td_mcts = TD_MCTS(iterations=4, rollout_depth=0, constant=np.sqrt(2), threshold_rate=0.8)

def get_action(state, score):
    init_model()

    legal_moves = get_legal_moves(state, score)
    legal_moves = filter_moves_by_td_threshold(state, score, legal_moves, threshold_rate=0.8)

    # legal_moves = filter_moves_keep_max_in_corner(state, legal_moves)
    # values = []
    # for a in legal_moves:
    #     temp_env = Game2048Env()
    #     temp_env.board = state.copy()
    #     temp_env.score = score
    #     temp_env.step(a, True)
    #     values.append(temp_env.score + tdl_estimate(temp_env.board))
    # action = legal_moves[np.argmax(values)]
    # return action

    global root
    if root is None:
        root = ChanceNode(state, score)
        root.calculate_untried_actions(td_mcts.threshold_rate)
    else:
        # assert(isinstance(root, AfterstateNode))
        find = False
        for c in root.children.values():
            if score == c.env.score and np.array_equal(state, c.env.board):
                root = c
                root.parent = None
                find = True
                break
        if not find:
            root = ChanceNode(state, score)
            root.calculate_untried_actions(td_mcts.threshold_rate)
    # assert(isinstance(root, ChanceNode))

    for _ in range(len(legal_moves)):
    # for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_act, s = td_mcts.best_action_distribution(root)

    # assert(len(root.untried_actions) == 0)
    # assert(len(root.children) == len(legal_moves))

    values = []
    for a in legal_moves:
        temp_env = Game2048Env()
        temp_env.board = state.copy()
        temp_env.score = score
        temp_env.step(a, True)
        values.append(temp_env.score + tdl_estimate(temp_env.board))
    children = []
    visits = []
    for a, c in root.children.items():
        children.append((a, c.total_reward / c.visits))
        visits.append((a, c.visits))
    children.sort()
    visits.sort()
    # values = np.array(values)
    # for i in range(len(legal_moves)):
    #     values[i] += children[i][1]
    # best_act = legal_moves[np.argmax(values)]

    t = '\t' if len(root.children) < 4 else ''
    tt = '\t' if len(root.children) < 3 else ''
    # print(f"{best_act}, {int(s.total_reward / s.visits)}, {[int(c.visits) for c in root.children.values()]}\t{t}{score}\t{int(root.total_reward / root.visits)}\t{[int(c.total_reward / c.visits) for c in root.children.values()]}")
    print(f"{[int(v) for _, v in visits]}\t{t}{score}\t{int(root.total_reward / root.visits)}\t{[int(v) for _, v in children]}\t{t}{tt}{[int(v) for v in values]}")

    root = root.children[best_act]
    # assert(isinstance(root, AfterstateNode))

    return best_act

if __name__ == "__main__":
    env = Game2048Env()
    env.reset()
    done = False
    while not done:
        legal_moves = env.legal_moves()
        if not legal_moves:
            break
        # action = get_action(env.board, env.score)
        # action = get_tdl_action(env.board)
        # legal_moves = filter_moves_keep_max_in_corner(env.board, legal_moves)
        values = []
        for a in legal_moves:
            temp_env = Game2048Env()
            temp_env.board = env.board.copy()
            temp_env.score = env.score
            temp_env.step(a, True)
            values.append(temp_env.score + tdl_estimate(temp_env.board))
        action = legal_moves[np.argmax(values)]
        _, _, done, _ = env.step(action)
    # env.render()
    print(f"Game Over! Final Score: {env.score}")
