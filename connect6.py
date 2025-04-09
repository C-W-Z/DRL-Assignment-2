import sys
import numpy as np
import random
from numba import njit
import copy
from operator import itemgetter

@njit
def _evaluate_position(board, size, r, c, color):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    score = 0
    opponent = 3 - color
    for dr, dc in directions:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 6:
            score += 1000
        elif count == 5:
            score += 500
        elif count == 4:
            score += 100
        elif count == 3:
            score += 10
        elif count == 2:
            score += 1

        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == opponent:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == opponent:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 6:
            score += 1000
        elif count == 5:
            score += 600
        elif count == 4:
            score += 120
        elif count == 3:
            score += 10
        elif count == 2:
            score += 1

    return score

@njit
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

@njit
def _availables(size, board):
    return [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]

@njit
def _occupies(size, board):
    return [(r, c) for r in range(size) for c in range(size) if board[r, c] != 0]

@njit
def _check_win(board, size):
    """Optimized check_win function using JIT."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                current_color = board[r, c]
                for dr, dc in directions:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == current_color:
                        continue
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == current_color:
                        count += 1
                        rr += dr
                        cc += dc
                    if count >= 6:
                        return current_color
    return 0

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.winner = 0
        self.last_opponent_move = None
        self.move_count = 0
        self.turn_moves = 0
        # self.mcts = MCTS(10, 1000)

    def reset_board(self):
        """Resets the board and game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.move_count = 0
        self.turn_moves = 0
        # self.mcts = MCTS(10, 1000)
        print("= ", flush=True)

    def set_board_size(self, size):
        """Changes board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.move_count = 0
        self.turn_moves = 0
        # self.mcts = MCTS(10, 1000)
        print("= ", flush=True)

    def availables(self):
        return _availables(self.size, self.board)

    def occupies(self):
        return _occupies(self.size, self.board)

    def do_move(self, move):
        r, c = move
        self.board[r, c] = self.turn
        self.move_count += 1
        self.turn_moves += 1
        if (self.move_count == 1 and self.turn == 1) or self.turn_moves == 2:
            self.turn = 3 - self.turn
            self.turn_moves = 0

    def game_end(self):
        self.check_win(True)
        return self.game_over, self.winner

    def check_win(self, real_play=False):
        """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
        if self.game_over:
            return self.winner
        self.winner = _check_win(self.board, self.size)
        if real_play and (self.winner != 0 or not self.availables()):
            self.game_over = True
        return self.winner

    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Processes a move and updates the board."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
                print("? Invalid move")
                return
            positions.append((row, col))

        self.last_opponent_move = positions[-1]  # Track the opponent's last move
        # self.mcts.update_with_move(positions[-1])

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

            self.move_count += 1
            self.turn_moves += 1
            if (self.move_count == 1 and color.upper() == 'B') or self.turn_moves == 2:
                self.turn = 3 - self.turn
                self.turn_moves = 0

        # self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates the best move based on predefined rules and ensures output."""
        self.check_win(True)
        if self.game_over:
            print("? Game over", flush=True)
            return

        # my_color = 1 if color.upper() == 'B' else 2
        # opponent_color = 3 - my_color

        # Winning move
        # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        # for r, c in empty_positions:
        #     self.board[r, c] = my_color
        #     if self.check_win() == my_color:
        #         self.board[r, c] = 0
        #         move_str = f"{self.index_to_label(c)}{r+1}"
        #         self.play_move(color, move_str)
        #         print(f"Rule 1, {move_str}", file=sys.stderr)
        #         print(move_str, flush=True)
        #         return
        #     self.board[r, c] = 0

        # expected_moves = 1 if self.move_count == 0 and color.upper() == 'B' else 2
        # if self.turn_moves >= expected_moves:
        #     print("? Turn already completed, wait for opponent's move")
        #     return

        mcts = MCTS(10, 1000)
        move = mcts.get_move(self)
        print([c.visits for c in mcts.root.children.values()], file=sys.stderr)
        print(f"move={move}, size={self.size}, {self.game_over}", file=sys.stderr)
        # mcts.update_with_move(move)
        if move:
            r, c = move
            move_str = f"{self.index_to_label(c)}{r+1}"
            self.play_move(color, move_str)
            print(move_str, flush=True)

    def evaluate_position(self, r, c, color):
        """Evaluates the strength of a position based on alignment potential."""
        return _evaluate_position(self.board, self.size, r, c, color)

    def show_board(self):
        """Displays the board in text format."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

def policy_value_func(game: Connect6Game, max_distance=3):
    # 獲取所有空位
    all_availables = game.availables()
    if not all_availables:
        return [], 0

    # 獲取已有棋子位置
    occupied = game.occupies()
    if not occupied:  # 如果棋盤為空，返回所有位置（初始情況）
        action_probs = np.ones(len(all_availables)) / len(all_availables)
        return all_availables, action_probs

    # 過濾距離戰場太遠的行動
    nearby_availables = []
    for pos in all_availables:
        # 檢查該位置是否在任何已有棋子的 max_distance 範圍內
        for occ in occupied:
            if manhattan_distance(pos, occ) <= max_distance:
                nearby_availables.append(pos)
                break  # 一旦滿足條件，跳出內層迴圈

    # 如果沒有附近位置，退回到所有可用位置（避免空列表）
    if not nearby_availables:
        nearby_availables = all_availables

    # 均勻分配概率給附近位置
    action_probs = np.ones(len(nearby_availables)) / len(nearby_availables)
    return nearby_availables, action_probs

# def rollout_policy_func(game: Connect6Game, nearby_availables):
#     """基於 evaluate_position 的啟發式 rollout 策略"""
#     if not nearby_availables:
#         nearby_availables = game.availables()
#         if not nearby_availables:
#             return []

#     scores = []
#     for r, c in nearby_availables:
#         # 臨時放置棋子並評估
#         game.board[r, c] = game.turn
#         score = game.evaluate_position(r, c, game.turn)
#         game.board[r, c] = 0
#         scores.append(score)

#     # 將分數轉換為概率（簡單正規化）
#     scores = np.array(scores)
#     if scores.max() == 0:  # 如果所有分數為 0，均勻分佈
#         probs = np.ones(len(nearby_availables)) / len(nearby_availables)
#     else:
#         probs = scores / scores.sum()  # 正規化為概率

#     return zip(nearby_availables, probs)

def get_rule_base_action(game: Connect6Game, nearby_availables):
    my_color = game.turn
    opponent_color = 3 - my_color
    # empty_positions = game.availables()

    for r, c in nearby_availables:
        game.board[r, c] = my_color
        if game.check_win() == my_color:
            game.board[r, c] = 0
            return (r, c)
        game.board[r, c] = 0

    for r, c in nearby_availables:
        game.board[r, c] = opponent_color
        if game.check_win() == opponent_color:
            game.board[r, c] = 0
            return (r, c)
        game.board[r, c] = 0

    scores = []
    best_move = None
    best_score = -1
    for r, c in nearby_availables:
        score = game.evaluate_position(r, c, my_color)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_move = (r, c)

    if max(scores) == 0:  # 如果所有分數為 0，均勻分佈
        return random.choice(nearby_availables)
    else:
        return best_move

def rollout_policy_func(game: Connect6Game, nearby_availables):
    """模拟神经网络随机生成各个节点的胜率P"""
    # rollout randomly
    # availables = game.availables()
    action_probs = np.random.rand(len(nearby_availables))
    return zip(nearby_availables, action_probs)

class MCTSNode:
    def __init__(self, parent=None, prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0
        self.P = prob # 從上一Node走到這一Node的機率
        self.untried_actions = []

    def expand(self, nearby_availables, action_probs):
        for action, prob in zip(nearby_availables, action_probs):
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    def select_child(self, c_puct):
        return max(self.children.items(), key=lambda c: c[1].get_ucb(c_puct))

    def update(self, leaf_value):
        self.visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.visits


    def get_ucb(self, exploration_constant):
        return self.Q + (exploration_constant * self.P * np.sqrt(self.parent.visits) / (1 + self.visits))

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, c_puct=5, n_playout=2000):
        self.root = MCTSNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, state: Connect6Game):
        # select
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select_child(self.c_puct)
            state.do_move(action)

        # print("select", file=sys.stderr)

        # expand
        nearby_availables, action_probs = policy_value_func(state)
        end, winner = state.game_end()
        if not end:
            node.expand(nearby_availables, action_probs)

        # print("expand", file=sys.stderr)

        # rollout
        leaf_value = self.rollout(state, nearby_availables)

        # print("rollout", file=sys.stderr)

        # backpropagation
        while node is not None:
            node.update(leaf_value)
            leaf_value = -leaf_value
            node = node.parent

        # print("backpropagation", file=sys.stderr)

    def rollout(self, state: Connect6Game, nearby_availables, limit=400):
        player = state.turn
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            # action_probs = rollout_policy_func(state, nearby_availables)
            # max_action = max(action_probs, key=itemgetter(1))[0]
            # print("get_rule_base_action", file=sys.stderr)
            max_action = get_rule_base_action(state, nearby_availables)
            # print(f"{max_action}", file=sys.stderr)
            state.do_move(max_action)
        # else:
        #     print("WARNING: rollout reached move limit", file=sys.stderr)
        #     return 0  # 如果達到限制，返回平局
        # print(f"winner: {winner}, {state.game_over}", file=sys.stderr)
        if winner == 0:  # Tie
            return 0
        return 1 if winner == player else -1

    def get_move(self, state: Connect6Game):
        for n in range(self.n_playout):
            # state_copy = Connect6Game(state.size)
            # state_copy.size = state.size
            # state_copy.board = state.board
            # state_copy.turn = state.turn
            # state_copy.game_over = state.game_over
            # state_copy.winner = state.winner
            # state_copy.last_opponent_move = state.last_opponent_move
            # state_copy.move_count = state.move_count
            # state_copy.turn_moves = state.turn_moves
            state_copy = copy.deepcopy(state)
            # print(f"playout {n}", file=sys.stderr)
            self.playout(state_copy)
        _max = max(self.root.children.items(), key=lambda c: c[1].visits)
        _min = min(self.root.children.values(), key=lambda c: c.visits)
        print(f"{_max[1].visits}, {_min.visits}", file=sys.stderr)
        if _max[1].visits - _min.visits <= 1:
            return random.choice(list(self.root.children.keys()))
        return _max[0]

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, 1.0)

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
