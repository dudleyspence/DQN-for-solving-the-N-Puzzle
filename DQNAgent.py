"""
Author: Dudley Spence
Title: Solving the N-puzzle using Deep Reinforcement Learning

DQNAgent
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import random
import os


class DQNAgent:
    """
    The agent that explores and learns to solve the puzzle. Learns to approximate the
    Q function allowing it to predict the actions that give the most cumulative reward
    """
    def __init__(self, final_epsilon=0.2, initial_epsilon=0.9, n=3, difficulty=4, nodes=250, gamma=0.9, learning_rate=0.001, summary=False):
        self.n = n
        self.N = n ** 2
        self.learning_rate = learning_rate
        self.difficulty = difficulty
        self.nodes = nodes
        self.checkpoint_path = "training_" + str(self.N-1) + "_puzzle_" + str(self.nodes) + "_nodes/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.q_net = self.build_dqn_model()
        if summary:
            self.q_net.summary()
        try:
            self.q_net.load_weights(self.checkpoint_path).expect_partial()
            self.epsilon = final_epsilon
        except Exception:
            self.epsilon = initial_epsilon
            pass
        self.target_q_net = self.build_dqn_model()
        self.gamma = gamma

    def build_dqn_model(self):
        """
        Builds a deep neural net which predicts the Q-values for all the possible actions given a state.
        The input has the same shape as the state, and the output should have the same shape as the action space
        since we want one Q-value per action.

        :return: Q network
        """
        input = self.N * (self.N-1)
        q_net = Sequential()
        q_net.add(Dense(self.nodes, input_shape=(input,), activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(self.nodes, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(self.nodes, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(4, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

        return q_net

    def epsilon_greedy_policy(self, state):
        """
        The epsilon greedy policy, either selecting an action using the policy network
        or a randomly selected action.
        :param state: The state to be used by the policy
        :return: action: number from 0 to 3
        """
        if np.random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return self.policy(state)

    def convert_board_one_hot(self, state):
        """
        Converts the categorical state data and one-hot encodes it then reshapes it.
        :param state: 1-D array with shape (1, N)
        :return: state: 1-D array with shape (1, N(N-1))
        """
        board = np.zeros((self.N, (self.N - 1)))
        for i in range(self.N):
            if state[i] != 0:
                num = state[i] - 1
                board[i][num] = 1
        board = tf.reshape(tf.convert_to_tensor(board, dtype='int64'), (self.n, self.n, self.N - 1))
        board = tf.reshape(board, (-1))
        return board

    def policy(self, state):
        """
        Forward propagates the state through the policy network and returns an action that has the highest
        Q-value.

        :param state: the state to be forward propagated
        :return: the calculated optimal action
        """
        action_q = self.q_net(state)
        action_q = action_q[0].numpy()
        optimal_action = np.argmax(action_q)
        return optimal_action

    def update_target_network(self):
        """
        Updates the parameters of the current target_q_net with the parameter of the current q_net.
        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to help optimise the
        approximation of the Q-function resulting in better predicted Q-values. Each epoch the
        parameters of the network are saved.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0)
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = batch
        current_q = self.q_net(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for i in range(len(state_batch)):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += self.gamma * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0, callbacks=[cp_callback])
        loss = training_history.history['loss']
        return loss
