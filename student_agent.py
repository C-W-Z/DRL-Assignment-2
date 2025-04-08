import numpy as np
from Game2048Env import Game2048Env
from TDL2048 import board, learning, pattern
import gc, atexit

def numpy_to_bitboard(np_board):
    b = board()
    for i in range(4):
        for j in range(4):
            value = np_board[i, j]
            if value != 0:
                tile = int(np.log2(value))
                b.set(i * 4 + j, tile)
    return b

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

def get_action(state, score):
    init_model()
    bitboard_state = numpy_to_bitboard(state)
    best_move = tdl.select_best_move(bitboard_state)
    tdl_action = best_move.action()
    return tdl_action_to_gym_action(tdl_action)

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
