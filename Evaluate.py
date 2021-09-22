"""
Author: Dudley Spence
Title: Solving the N-puzzle using Deep Reinforcement Learning

Evaluate

Cycles through each difficulty attempting to solve 100 puzzles to determine the rate of solved
for each difficulty. The agent is given 50 actions to solve each puzzle.
"""
from Environment import Environment
from DQNAgent import DQNAgent
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def test(board, N_attempts, agent, env):
    """

    :param board: the state to be tested
    :param N_attempts: the number of actions given to solve each puzzle
    :param agent: the instance of the agent class
    :param env: the instance of the environment
    :return: done: Boolean
    """
    N = len(board)
    n = int(math.sqrt(N))
    env.current_state = board.copy()
    for i in range(N_attempts):
        state = agent.convert_board_one_hot(env.current_state)
        state = tf.reshape(state, (1, -1))
        action = agent.policy(state)
        next_state, reward, done = env.apply_action(action)
        if done:
            break
    return done


def evaluate(n):
    """
    Evaluates the progression of the network by giving it 100 puzzles to solve at each difficulty
    and graphs the rate of success at each difficulty.
    :param n: the number of tiles per row and column
    :return: None
    """
    env = Environment(n)
    solveRates = []
    x = []
    agent = DQNAgent(n=n)
    if n == 3:
        maxOptimum = 22
    else:
        maxOptimum = 56

    for i in tqdm(range(1, maxOptimum)):
        solves = 0
        x.append(i)
        for j in range(100):
            puzzle = env.generate_new_board(i)
            result = test(puzzle, 50, agent, env)
            if result:
                solves += 1
            else:
                continue
        solveRate = solves
        solveRates.append(solveRate)
    solveRates = np.array(solveRates)

    plt.plot(x, solveRates)
    plt.title('Rate of solved puzzle for different difficulties')
    plt.xlabel('Manhattan Distance')
    plt.ylabel('Rate of solved')
    plt.show()


N = int(input("For the N-puzzle you would like to evaluate, what is the N? "))
n = int(math.sqrt(N+1))
print(n)
evaluate(n)