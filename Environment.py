"""
Author: Dudley Spence
Title: Solving the N-puzzle using Deep Reinforcement Learning

Environment
"""
import random
import numpy as np


class Environment:
    """
    The Environment maintains the current state of the environment and calculates the
    reward of each state transition.
    """

    def __init__(self, n):
        self.n = n
        self.N = n**2
        self.solved_board = list(range(0, self.N, 1))
        self.current_state = self.solved_board[:]
        self.possible_action = True

    def swap_tiles(self, pos1, pos2):
        """
        Takes two tile positions and swaps them in the state array
        :param pos1: The tile moving into vacant position
        :param pos2: Always tile 0 (vacant position)
        :return: None
        """
        self.current_state[pos1], self.current_state[pos2] = self.current_state[pos2], self.current_state[pos1]
        return

    @staticmethod
    def gen_rand_action():
        """
        Generates a random action
        :return: action: integer from 0 to 3
        """
        n = random.randint(0, 3)
        return n

    def check_reward(self):
        """
        Calculates the total reward of a state using heuristics and whether the action
        attempted was successful
        :return: reward
        """
        reward = 0
        if not self.possible_action:
            reward -= 1
        reward -= self.manhattan_linear_conflict()
        return reward

    def manhattan_linear_conflict(self):
        """
        Calculates an estimate of the distance which is a combination of the heuristic
        manhattan distance and linear conflict.
        :return: Distance: an estimate of the minimum number of moves required to solve the puzzle
        """
        return self.manhattan_distance() + self.linear_conflict()

    def manhattan_distance(self):
        """
        Computes the total manhattan distance of a state by finding the sum of the taxicab
        distances between each tiles position and its goal position, ignoring tile 0.
        :return: The manhattan distance
        """
        def taxicabDistance(xA, yA, xB, yB):
            distance = abs(xA - xB) + abs(yA - yB)
            return distance

        initial_state = np.reshape(self.current_state, (self.n, self.n))
        goal_state = np.reshape(self.solved_board, (self.n, self.n))
        total_distance = 0

        for i in range(1, self.N):
            y1, x1 = np.where(initial_state == i)
            y2, x2 = np.where(goal_state == i)
            total_distance += taxicabDistance(x1[0], y1[0], x2[0], y2[0])

        return total_distance

    def linear_conflict(self):
        """
        Calculates the number of linear conflicts in a state and for each
        conflict adds +2 to the linear conflict score.
        :return: linear conflict score
        """
        goal_state = np.array(self.solved_board).reshape(self.n, self.n)
        linear_conflict_score = 0
        state = self.display_board()
        for i in range(0, self.n):
            for j in range(0, self.n - 1):
                current_num = state[i][j]
                adjacent_num = state[i][j + 1]
                goal_num = goal_state[i][j]
                goal_adjacent = goal_state[i][j + 1]
                if current_num != 0 and adjacent_num != 0:
                    if current_num == goal_adjacent and goal_num == adjacent_num:
                        linear_conflict_score += 2

        for i in range(0, self.n - 1):
            for j in range(0, self.n):
                current_num = state[i][j]
                adjacent_num = state[i + 1][j]
                goal_num = goal_state[i][j]
                goal_adjacent = goal_state[i + 1][j]
                if current_num != 0 and adjacent_num != 0:
                    if current_num == goal_adjacent and goal_num == adjacent_num:
                        linear_conflict_score += 2
        return linear_conflict_score

    def apply_action(self, action):
        """
        Applies an action to the board.
        :param action: The action provided
        :return: Next_state, reward, done
        """
        self.possible_action = True
        # UP
        if action == 0 and self.current_state.index(0) <= (self.N-(self.n+1)):
            self.swap_tiles(self.current_state.index(0) + self.n, self.current_state.index(0))
        # DOWN
        elif action == 1 and self.current_state.index(0) >= self.n:
            self.swap_tiles(self.current_state.index(0) - self.n, self.current_state.index(0))
        # LEFT
        elif action == 2 and ((self.current_state.index(0) + 1) % self.n != 0):
            self.swap_tiles(self.current_state.index(0) + 1, self.current_state.index(0))
        # RIGHT
        elif action == 3 and ((self.current_state.index(0) + 1) % self.n != 1):
            self.swap_tiles(self.current_state.index(0) - 1, self.current_state.index(0))
        else:
            self.possible_action = False

        reward = self.check_reward()

        if self.current_state == self.solved_board:
            done = True
            reward += 10
        else:
            done = False

        return self.current_state, reward, done

    def generate_new_board(self, difficulty=10):
        """
        Generates a board by repeatedly making random actions until the calculated distance
        is greater than the difficulty
        :param: difficulty
        :return: The current state
        """
        self.current_state = self.solved_board[:]
        distance = 0
        while distance < difficulty:
            random_action = self.gen_rand_action()
            self.apply_action(random_action)
            distance = self.manhattan_linear_conflict()
        return self.current_state

    def display_board(self):
        """
        Displays the board as a 2D-array
        :return: The display board
        """
        display = np.array(self.current_state).reshape(self.n, self.n)
        return display





