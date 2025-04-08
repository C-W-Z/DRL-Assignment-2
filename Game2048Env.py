import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from numba import njit

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

    def legal_moves(self):
        return get_legal_moves(self.board, self.score)
