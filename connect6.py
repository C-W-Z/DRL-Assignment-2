import sys
import numpy as np
import random
from numba import njit
import copy
from operator import itemgetter

def my_excepthook(exctype, value, traceback):
    sys.stderr.write(f"未處理的異常: {value}\n")
    sys.__excepthook__(exctype, value, traceback)  # 保留原始異常處理

sys.excepthook = my_excepthook

@njit
def evaluate_board(board: np.ndarray, size: int, color: int) -> tuple:
    """
    評估6子棋盤面，1為黑子，2為白子，0為空格
    board: 二維NumPy陣列表示棋盤
    size: 棋盤大小
    color: 當前玩家 (1 或 2)
    返回: (己方分數, 對方分數)
    """
    # 分數表轉為陣列 (索引對應連子數量)
    score_array = np.array([0, 1, 50, 500, 5000, 10000, 1000000], dtype=np.int64)

    opponent = 3 - color
    my_score = 0
    oppo_score = 0

    # 中心位置權重 (預先計算為NumPy陣列)
    # center_weight = np.zeros((size, size), dtype=np.int32)
    # for i in range(size):
    #     for j in range(size):
    #         center_weight[i, j] =min(i, size-1-i) + min(j, size-1-j) + 1

    # 方向陣列
    directions = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)

    for i in range(size):
        for j in range(size):
            # 添加中心位置分數
            # if board[i, j] == color:
            #     my_score += center_weight[i, j]
            # elif board[i, j] == opponent:
            #     oppo_score += center_weight[i, j]

            # print(f"evaluate {i}, {j}", file=sys.stderr)

            # 檢查每個方向的6格窗口
            for direction in directions:
                di, dj = direction[0], direction[1]
                if can_fit_window(i, j, di, dj, size):
                    window = get_window(board, i, j, di, dj, size)
                    my_score += analyze_pattern(window, color, score_array)
                    oppo_score += analyze_pattern(window, opponent, score_array)

    return my_score, oppo_score

@njit
def can_fit_window(i: int, j: int, di: int, dj: int, size: int, length: int = 6) -> bool:
    """檢查是否能在該方向放入6格窗口"""
    ni, nj = i + di * (length - 1), j + dj * (length - 1)
    return 0 <= ni < size and 0 <= nj < size

@njit
def get_window(board: np.ndarray, i: int, j: int, di: int, dj: int, size: int) -> np.ndarray:
    """獲取指定方向的6格窗口"""
    window = np.zeros(6, dtype=np.int8)
    for k in range(6):
        window[k] = board[i + k * di, j + k * dj]
    return window

@njit
def analyze_pattern(window: np.ndarray, player: int, score_array: np.ndarray) -> int:
    """
    分析6格窗口的模式，包括連子、開放端和潛在威脅
    """
    score = 0
    opponent = 3 - player

    # 計算連續棋子數量和相關特徵
    max_consecutive = 0
    current_consecutive = 0
    open_ends = 0
    player_count = 0
    empty_count = 0
    opponent_count = 0

    # 檢查窗口前的空格
    if window[0] == 0:
        open_ends += 1

    # 分析窗口
    for i in range(6):
        if window[i] == player:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
            player_count += 1
        elif window[i] == opponent:
            current_consecutive = 0
            opponent_count += 1
        else:  # 空格
            if current_consecutive > 0 and i < 5 and window[i + 1] == player:
                open_ends += 1
            current_consecutive = 0
            empty_count += 1

    # 檢查窗口後的開放端
    if window[-1] == player and current_consecutive > 0:
        open_ends += 1

    if opponent_count == 0:

        if max_consecutive > 1:
            score += open_ends * max_consecutive

        score += score_array[player_count]
        # if player_count == 2 and empty_count == 4 and max_consecutive == 1:
        #     score += score_array[2]
        # if player_count == 3 and empty_count == 3:
        #     score += score_array[3] * 0.8
        # if player_count == 4 and empty_count == 2:
        #     score += score_array[4] * 0.8
        # if player_count == 5 and empty_count == 1:
        #     score += score_array[5] * 0.8

    return score

@njit
def _evaluate_position(board, size, r, c, color):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    score = 5
    if r == 0 or r == size - 1 or c == 0 or c == size - 1:
        score = 0
    opponent = 3 - color

    for dr, dc in directions:
        count = 1  # 包括當前位置
        forward_end = r + dr
        backward_end = r - dr
        forward_cc = c + dc
        backward_cc = c - dc

        # 正方向連子
        rr, cc = forward_end, forward_cc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr += dr
            cc += dc
        forward_end = rr  # 正向終點
        forward_cc = cc

        # 反方向連子
        rr, cc = backward_end, backward_cc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr -= dr
            cc -= dc
        backward_end = rr  # 反向終點
        backward_cc = cc

        forward_open = (0 <= forward_end < size and 0 <= forward_cc < size and board[forward_end, forward_cc] == 0)
        backward_open = (0 <= backward_end < size and 0 <= backward_cc < size and board[backward_end, backward_cc] == 0)

        forward_space = 0
        forward_count = 0
        if forward_open:
            rr, cc = forward_end, forward_cc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] != opponent and forward_space < 5:
                forward_space += 1 if board[rr, cc] == 0 else 0
                if board[rr, cc] == color and (forward_space == 1 or forward_count == 1 and forward_space == 2):
                    forward_count += 1
                rr += dr
                cc += dc

        backward_space = 0
        backward_count = 0
        if backward_open:
            rr, cc = backward_end, backward_cc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] != opponent and backward_space < 5:
                backward_space += 1 if board[rr, cc] == 0 else 0
                if board[rr, cc] == color and (backward_space == 1 or backward_count == 1 and backward_space == 2):
                    backward_count += 1
                rr -= dr
                cc -= dc

        count += forward_count + backward_count
        total_space = forward_space + backward_space
        can_reach_six = count + total_space >= 6

        if not can_reach_six:
            continue

        s = 0
        if count >= 6:
            s += 10000
        elif count == 5:
            s += 1000
        elif count == 4:
            s += 500
        elif count == 3:
            s += 200
        elif count == 2:
            s += 50
        score += s

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
def _nearby_availables(size, board, max_dist=2):
    res = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
    for r in range(size):
        for c in range(size):
            if board[r, c] == 0:
                continue
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < size and 0 <= cc < size):
                    break
                if board[rr, cc] == 0:
                    res.append((rr, cc))
                for _ in range(max_dist - 1):
                    rr, cc = rr + dr, cc + dc
                    if not (0 <= rr < size and 0 <= cc < size):
                        break
                    if board[rr, cc] == 0:
                        res.append((rr, cc))
    return res

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

@njit
def winning_move(board: np.ndarray, piece: int) -> bool:
    """
    檢查是否有6連勝的情況
    board: 二維NumPy陣列表示棋盤 (1為黑子，2為白子，0為空格)
    piece: 要檢查的棋子 (1 或 2)
    返回: True表示該棋子獲勝，False表示未獲勝
    """
    WINDOW_LENGTH = 6
    rows, cols = board.shape[0], board.shape[1]

    # 檢查水平方向
    for r in range(rows):
        for c in range(cols - WINDOW_LENGTH + 1):
            window = board[r, c:c + WINDOW_LENGTH]
            if np.all(window == piece):
                return True

    # 檢查垂直方向
    for c in range(cols):
        for r in range(rows - WINDOW_LENGTH + 1):
            window = board[r:r + WINDOW_LENGTH, c]
            if np.all(window == piece):
                return True

    # 檢查正斜線 (左上到右下)
    for r in range(rows - WINDOW_LENGTH + 1):
        for c in range(cols - WINDOW_LENGTH + 1):
            # 提取對角線元素
            diagonal = np.zeros(WINDOW_LENGTH, dtype=np.int8)
            for i in range(WINDOW_LENGTH):
                diagonal[i] = board[r + i, c + i]
            if np.all(diagonal == piece):
                return True

    # 檢查負斜線 (右上到左下)
    for r in range(WINDOW_LENGTH - 1, rows):
        for c in range(cols - WINDOW_LENGTH + 1):
            # 提取對角線元素
            diagonal = np.zeros(WINDOW_LENGTH, dtype=np.int8)
            for i in range(WINDOW_LENGTH):
                diagonal[i] = board[r - i, c + i]
            if np.all(diagonal == piece):
                return True

    return False

@njit
def minimax(board: np.ndarray, size: int, depth: int, alpha: float, beta: float,
            maximizingPlayer: bool, maximizingColor: int) -> tuple:
    """
    Minimax演算法，帶Alpha-Beta剪枝，用於6子棋
    board: 二維NumPy陣列表示棋盤
    size: 棋盤大小
    depth: 搜尋深度
    alpha, beta: Alpha-Beta剪枝參數
    maximizingPlayer: 是否為最大化玩家
    maximizingColor: 最大化玩家的棋子 (1 或 2)
    返回: (action, value)，action是(row, col)或(-1, -1)表示無動作
    """
    # 獲取有效位置
    valid_locations = _nearby_availables(size, board)
    n_valid = len(valid_locations)

    # 檢查終止條件
    winner = _check_win(board, size)
    is_terminal = winner != 0 or n_valid == 0

    if depth == 0 or is_terminal:
        if is_terminal:
            if winner == maximizingColor:
                return (-1, -1), 100000000
            elif winner == 3 - maximizingColor:
                return (-1, -1), -100000000
            else:  # 平局
                return (-1, -1), 0
        else:  # 深度為0
            my_score, oppo_score = evaluate_board(board, size, maximizingColor)
            return (-1, -1), my_score - oppo_score

    if maximizingPlayer:
        value = -np.inf
        action = (-1, -1)  # 預設動作
        if n_valid > 0:
            # 隨機選擇初始動作（避免random.choice）
            action = valid_locations[0]

        for i in range(n_valid):
            row, col = valid_locations[i]
            b_copy = board.copy()
            b_copy[row, col] = maximizingColor
            _, new_score = minimax(b_copy, size, depth - 1, alpha, beta, False, maximizingColor)
            if new_score > value:
                value = new_score
                action = (row, col)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return action, value

    else:  # Minimizing player
        value = np.inf
        action = (-1, -1)  # 預設動作
        if n_valid > 0:
            action = valid_locations[0]

        for i in range(n_valid):
            row, col = valid_locations[i]
            b_copy = board.copy()
            b_copy[row, col] = 3 - maximizingColor
            _, new_score = minimax(b_copy, size, depth - 1, alpha, beta, True, maximizingColor)
            if new_score < value:
                value = new_score
                action = (row, col)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return action, value

class Board:
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

    def nearby_availables(self, max_dist=2):
        return _nearby_availables(self.size, self.board, max_dist)

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

    def evaluate_position(self, r, c, color):
        """Evaluates the strength of a position based on alignment potential."""
        return _evaluate_position(self.board, self.size, r, c, color)

# def policy_value_func(game: Board, max_distance=2):
#     # 獲取所有空位
#     all_availables = game.availables()
#     if not all_availables:
#         return [], 0
#     if len(all_availables) == game.size ** 2:
#         action_probs = np.ones(len(all_availables)) / len(all_availables)
#         return all_availables, action_probs

#     # 過濾距離戰場太遠的行動
#     nearby_availables = game.nearby_availables(max_distance)

#     # 如果沒有附近位置，退回到所有可用位置（避免空列表）
#     if not nearby_availables:
#         nearby_availables = all_availables

#     # 均勻分配概率給附近位置
#     action_probs = np.ones(len(nearby_availables)) / len(nearby_availables)
#     return nearby_availables, action_probs

def rollout_policy_func(game: Board, threshold_rate = 0, max_dist=2):
    """基於 evaluate_position 的啟發式 rollout 策略"""
    nearby_availables = game.nearby_availables(max_dist)
    if not nearby_availables:
        nearby_availables = game.availables()
        if not nearby_availables:
            return [], np.array([])

    scores = []
    for r, c in nearby_availables:
        score = game.evaluate_position(r, c, game.turn) + game.evaluate_position(r, c, 3 - game.turn)
        scores.append(score)

    # 將分數轉換為概率（簡單正規化）
    scores = np.array(scores)
    if scores.max() == 0:  # 如果所有分數為 0，均勻分佈
        probs = np.ones(len(_nearby_availables)) / len(_nearby_availables)
    else:
        # softmax
        scores = np.exp(scores / 100)
        # probs = (scores) / scores.sum()

    threshold_prob = max(scores) * threshold_rate
    _nearby_availables = []
    _scores = []
    total_score = 0.0
    for action, score in zip(nearby_availables, scores):
        if score < threshold_prob:
            continue
        _nearby_availables.append(action)
        _scores.append(score)
        total_score += score

    _scores = np.array(_scores)
    probs = (_scores) / _scores.sum()

    return _nearby_availables, probs

# def rollout_policy_func(game: Board, nearby_availables):
#     """模拟神经网络随机生成各个节点的胜率P"""
#     # rollout randomly
#     # availables = game.availables()
#     action_probs = np.random.rand(len(nearby_availables))
#     return zip(nearby_availables, action_probs)

def get_rule_base_action(game: Board, nearby_availables):
    # print("get_rule_base_action", file=sys.stderr)
    my_color = game.turn
    opponent_color = 3 - my_color
    # empty_positions = game.availables()

    # print("check me win", file=sys.stderr)

    for r, c in nearby_availables:
        if not (0 <= r < game.size and 0 <= c < game.size):
            print(f"{r}, {c}", file=sys.stderr)
        game.board[r, c] = my_color
        if game.check_win() == my_color:
            game.board[r, c] = 0
            return (r, c)
        game.board[r, c] = 0

    # print("check oppo win", file=sys.stderr)

    for r, c in nearby_availables:
        game.board[r, c] = opponent_color
        if game.check_win() == opponent_color:
            game.board[r, c] = 0
            return (r, c)
        game.board[r, c] = 0

    # print("evaluate_position", file=sys.stderr)
    # scores = []
    best_move = None
    best_score = -1
    for r, c in nearby_availables:
        score = game.evaluate_position(r, c, my_color)
        # scores.append(score)
        if score > best_score:
            best_score = score
            best_move = (r, c)

    # print("evaluate_position 2", file=sys.stderr)
    for r, c in nearby_availables:
        score = game.evaluate_position(r, c, opponent_color)
        # scores.append(score)
        if score >= best_score:
            best_score = score
            best_move = (r, c)

    # print(f"best_move:{best_move}", file=sys.stderr)

    if best_score == 0:  # 如果所有分數為 0，均勻分佈
        return random.choice(nearby_availables)
    else:
        return best_move

class MCTSNode:
    def __init__(self, parent=None, prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0
        # self.P = prob # 從上一Node走到這一Node的機率
        self.P = 1.0
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
        if self.visits == 0:
            return np.inf
        return self.Q + (exploration_constant * self.P * np.sqrt(self.parent.visits) / (self.visits))

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, c_puct=5, n_playout=2000):
        self.root = MCTSNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, state: Board):
        # select
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select_child(self.c_puct)
            state.do_move(action)

        # print("select", file=sys.stderr)

        # expand
        nearby_availables, action_probs = rollout_policy_func(state, 0.05)
        end, winner = state.game_end()
        if not end:
            node.expand(nearby_availables, action_probs)
            action, node = node.select_child(self.c_puct)
            state.do_move(action)

        # print("expand", file=sys.stderr)

        # rollout
        leaf_value = self.rollout(state)

        # print("rollout", file=sys.stderr)

        # backpropagation
        while node is not None:
            node.update(leaf_value)
            leaf_value = -leaf_value
            node = node.parent

        # print("backpropagation", file=sys.stderr)

    def rollout(self, state: Board, limit=20):
        player = state.turn
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break

            empty_positions = state.availables()
            for r, c in empty_positions:
                state.board[r, c] = state.turn
                if state.check_win() == state.turn:
                    state.board[r, c] = 0
                    state.do_move((r, c))
                state.board[r, c] = 0

            nearby_availables, probs = rollout_policy_func(state, 0.01, 2)
            # action = max(action_probs, key=itemgetter(1))[0]
            action = np.random.choice(len(nearby_availables), p=probs.ravel())
            action = nearby_availables[action]
            # action = get_rule_base_action(state, nearby_availables)
            # print(f"{action}", file=sys.stderr)
            state.do_move(action)
        else:
        #     print("WARNING: rollout reached move limit", file=sys.stderr)
            # print("evaluate_board", file=sys.stderr)
            my_score, oppo_score = evaluate_board(state.board, state.size, player)
            # print("evaluate_board end", file=sys.stderr)
            # if my_score > oppo_score:
            #     return 1
            # elif my_score < oppo_score:
            #     return -1
            # else:
            #     return 0
            score = 0
            empty_positions = state.nearby_availables()
            for r, c in empty_positions:
                score += state.evaluate_position(r, c, player)
                score -= state.evaluate_position(r, c, 3 - player)
            if score > 0 and my_score > oppo_score:
                return 1
            elif score < 0 and my_score < oppo_score:
                return -1
            return 0  # 如果達到限制，返回平局
        # print(f"winner: {winner}, {state.game_over}", file=sys.stderr)
        if winner == 0:  # Tie
            return 0
        return 1 if winner == player else -1

    def get_move(self, state: Board):
        for n in range(self.n_playout):
            if len(self.root.children) == 1:
                _max = max(self.root.children.items(), key=lambda c: c[1].visits)
                print(f"{_max[0]}", file=sys.stderr)
                return _max[0]
            # state_copy = Board(state.size)
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
        # print("update_with_move", file=sys.stderr)
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, 1.0)

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
        self.mcts = MCTS(1.41, 1000)

    def reset_board(self):
        """Resets the board and game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.move_count = 0
        self.turn_moves = 0
        self.mcts = MCTS(1.41, 1000)
        print("= ", flush=True)

    def set_board_size(self, size):
        """Changes board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.move_count = 0
        self.turn_moves = 0
        self.mcts = MCTS(1.41, 1000)
        print("= ", flush=True)

    def availables(self):
        return _availables(self.size, self.board)

    def occupies(self):
        return _occupies(self.size, self.board)

    def nearby_availables(self, max_dist=2):
        return _nearby_availables(self.size, self.board, max_dist)

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
        self.mcts.update_with_move(positions[-1])

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

            self.move_count += 1
            self.turn_moves += 1
            if (self.move_count == 1 and color.upper() == 'B') or self.turn_moves == 2:
                self.turn = 3 - self.turn
                self.turn_moves = 0

        # print("evaluate_board", file=sys.stderr)
        black_score, white_score = evaluate_board(self.board, self.size, 1)
        print(f"B: {black_score}, W: {white_score}", file=sys.stderr)

        black_score = 0
        white_score = 0
        empty_positions = self.nearby_availables()
        # print(empty_positions, file=sys.stderr)
        for r, c in empty_positions:
            black_score += self.evaluate_position(r, c, 1)
            white_score += self.evaluate_position(r, c, 2)
        print(f"B: {black_score}, W: {white_score}", file=sys.stderr)
        print("------", file=sys.stderr)

        # self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates the best move based on predefined rules and ensures output."""
        self.check_win(True)
        if self.game_over:
            print("? Game over", flush=True)
            return

        if self.move_count == 0 and color.upper() == 'B':
            sigma = self.size / 6  # 預設sigma為棋盤大小的1/6，使分佈適中

            # 定義棋盤中心的均值 (mu_x, mu_y)
            mu = (self.size - 1) / 2  # 中心點，例如size=6時為2.5

            # 使用高斯分布生成座標
            while True:
                # 生成兩個獨立的高斯隨機變數
                x = np.random.normal(loc=mu, scale=sigma)
                y = np.random.normal(loc=mu, scale=sigma)

                # 四捨五入到整數並檢查是否在棋盤範圍內
                x_int = int(np.round(x))
                y_int = int(np.round(y))
                print(x_int, y_int, file=sys.stderr)

                if 0 <= x_int < self.size and 0 <= y_int < self.size:
                    move = (x_int, y_int)
                    break

            # move = np.random.choice([(r, c) for r in range(self.size) for c in range(self.size)], p=center_weight.ravel())
            r, c = move
            move_str = f"{self.index_to_label(c)}{r+1}"
            print(move, file=sys.stderr)
            self.play_move(color, move_str)
            print(move_str, flush=True)
            return

        my_color = 1 if color.upper() == 'B' else 2
        opponent_color = 3 - my_color
        empty_positions = self.availables()
        for r, c in empty_positions:
            self.board[r, c] = my_color
            if self.check_win() == my_color:
                self.board[r, c] = 0
                move_str = f"{self.index_to_label(c)}{r+1}"
                self.play_move(color, move_str)
                print(f"Rule 1, {move_str}", file=sys.stderr)
                print(move_str, flush=True)
                return
            self.board[r, c] = 0

        print("Minimax", file=sys.stderr)
        action, value = minimax(self.board, self.size, 2, -np.inf, np.inf, True, my_color)
        print(f"Minimax: {action}, {value}", file=sys.stderr)
        r, c = action
        move_str = f"{self.index_to_label(c)}{r+1}"
        print(f"move={move_str}, size={self.size}, {self.game_over}", file=sys.stderr)
        self.play_move(color, move_str)
        print(move_str, flush=True)
        return

        # expected_moves = 1 if self.move_count == 0 and color.upper() == 'B' else 2
        # if self.turn_moves >= expected_moves:
        #     print("? Turn already completed, wait for opponent's move")
        #     return

        b = Board(self.size)
        b.size = self.size
        b.board = self.board
        b.turn = self.turn
        b.game_over = self.game_over
        b.winner = self.winner
        b.last_opponent_move = self.last_opponent_move
        b.move_count = self.move_count
        b.turn_moves = self.turn_moves

        # r, c = get_rule_base_action(b, nearby)

        # nearby_availables, probs = rollout_policy_func(b)
        # # print(nearby_availables, probs, file=sys.stderr)
        # action = np.random.choice(len(nearby_availables), p=probs.ravel())
        # # print(f"{action}", file=sys.stderr)
        # r, c = nearby_availables[action]
        # move_str = f"{self.index_to_label(c)}{r+1}"
        # self.play_move(color, move_str)
        # print(move_str, flush=True)
        # return

        move = self.mcts.get_move(b)
        near, prob = rollout_policy_func(b, 0.05)
        print([c.visits for c in self.mcts.root.children.values()], file=sys.stderr)
        for a, c in self.mcts.root.children.items():
            move_str = f"{self.index_to_label(a[1])}{a[0]+1}"
            p = 0
            for i in range(len(near)):
                if near[i] == a:
                    p = prob[i]
                    break
            print(f"{move_str}, {c.visits}, {p}", file=sys.stderr)

        # mcts.update_with_move(move)
        if move:
            r, c = move
            move_str = f"{self.index_to_label(c)}{r+1}"
            print(f"move={move_str}, size={self.size}, {self.game_over}", file=sys.stderr)
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

if __name__ == "__main__":
    game = Connect6Game()

    # game.board[15, game.label_to_index("J")] = 1
    # game.board[15, game.label_to_index("K")] = 2
    # game.board[15, game.label_to_index("L")] = 2
    # game.board[15, game.label_to_index("N")] = 2
    # game.board[15, game.label_to_index("O")] = 2
    # print(game.evaluate_position(15, game.label_to_index("M"), 2), game.evaluate_position(15, game.label_to_index("M"), 1))
    # print(game.evaluate_position(15, game.label_to_index("P"), 2), game.evaluate_position(15, game.label_to_index("P"), 1))
    # print(game.evaluate_position(15, game.label_to_index("Q"), 2), game.evaluate_position(15, game.label_to_index("Q"), 1))
    # game.show_board()
    # game.generate_move('B')
    # game.show_board()
    # game.generate_move('W')
    # game.show_board()
    # game.generate_move('W')
    # game.show_board()
    # game.generate_move('B')
    # game.show_board()
    # game.generate_move('B')
    # game.show_board()

    game.run()
