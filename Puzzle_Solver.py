"""
Author: Dudley Spence
Title: Solving the N-puzzle using Deep Reinforcement Learning

Puzzle Solver

Takes a puzzle as input in 1-D format as a long number e.g. 012345678
uses the policy network to provide the user with a step-by-step solution.
"""
from Environment import Environment
from DQNAgent import DQNAgent
import math
import tensorflow as tf
import numpy as np


def check_solvable(board, n):
    """
    Checks if the given puzzle is solvable
    :param board: the user provided puzzle
    :param n: the number of tiles per column and row
    :return: Boolean
    """
    board_copy = board.copy()
    board_copy.remove(0)
    inversions = 0
    board = np.array(board)
    for i in range(len(board_copy)):
        for j in range(len(board_copy)-i):
            if board_copy[i] > board_copy[i+j]:
                inversions += 1
    inversions_even = bool(inversions % 2 == 0)
    if not (n % 2 == 0):
        if inversions_even:
            return True
        else:
            return False
    else:
        board = board.reshape(n, n)
        empty_spot = np.where(board == 0)[0][0]
        check_even = bool((n-empty_spot) % 2 == 0)
        if check_even:
            if not inversions_even:
                return True
            else:
                return False
        else:
            if inversions_even:
                return True
            else:
                return False


def test(board, N_attempts):
    """
    Takes a user provided puzzle state and uses the DQN to present a step-by-step solution.
    :param board: the puzzle state
    :param N_attempts: the number of actions given to solve the board
    :return: None
    """
    N = len(board)
    n = int(math.sqrt(N))
    if not check_solvable(board, n):
        print("This puzzle has no solvable solution")
        return False
    agent = DQNAgent(n=n)
    env = Environment(n)
    env.current_state = board
    optimum_moves = env.manhattan_linear_conflict()
    print('\n')
    print(f"The difficulty of this puzzle is: {optimum_moves}")
    moves_list = [env.display_board()]
    for i in range(N_attempts):
        state = agent.convert_board_one_hot(env.current_state)
        state = tf.reshape(state, (1, -1))
        action = agent.policy(state)
        next_state, reward, done = env.apply_action(action)
        moves_list.append(env.display_board())
        if done:
            print(f"This puzzle took {i+1} moves to solve")
            for j in range(len(moves_list)):
                print(moves_list[j])
                print('\n')
            break
    if not done:
        print(f"After {N_attempts} moves this puzzle was not solved")
        print('\n')


puzzle_input = input("Please enter the puzzle as a number of length N e.g. 012345678: ")

puzzle = [int(x) for x in str(puzzle_input)]

test(puzzle, 50)



