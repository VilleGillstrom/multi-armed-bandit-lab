# epsilon-greedy example implementation of a multi-armed bandit
import random

import os, sys, inspect
import numpy as np
import math

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import simulator
import reference_bandit


def normalize(lst, lower, upper):
    lmin = min(lst)
    lmax = max(lst)
    return [(upper - lower) * ((x - lmin) / (lmax - lmin)) + lower for x in lst]


# generic epsilon-greedy bandit
class Bandit:
    def __init__(self, arms, epsilon=0.2):
        self.arms = list(arms)  # The arms to be pulled
        self.epsilon = epsilon  # Epsilon to
        self.frequencies = [0] * len(arms)  # How many times arm has been pulled
        self.sums = [0] * len(arms)  # Total value created

        self.tau = 1  # temperature
        self.q_values = [0] * len(arms)
        self.discount = 0.9

        self.alpha = 0.8
        self.sigma = 15  # low values cause high exploration at value changes
        self.d = 1 / len(arms)  # inverse sensitivity (used for epsilon)



    def run(self):
        action = ""

        if min(self.frequencies) == 0:
            # Will pull each arm once at start
            arm_to_pull_index = self.frequencies.index(min(self.frequencies))
            action = self.arms[arm_to_pull_index]
        elif random.random() < self.epsilon:
            action = self.explore()
        else:
            action = self.exploit()
        return action

    def exploit(self):
        # Pick arm with the highest predicted reward
        arm_to_pull_index = self.q_values.index(max(self.q_values))
        action = self.arms[arm_to_pull_index]
        return action

    def explore(self):
        # normalize Q values to make e^q sensible
        q_norms = normalize(self.q_values, -1, 1)
        q_sum = sum([math.pow(math.e, v / self.tau) for v in q_norms])

        #Assign a probability to explore each action
        action_probabilities = [math.pow(math.e, (v / self.tau)) / q_sum for v in q_norms]

        action = np.random.choice(self.arms, 1, p=action_probabilities)
        return action

    def give_feedback(self, arm, reward):
        arm_index = self.arms.index(arm)

        # Total Value
        sum = self.sums[arm_index] + reward
        # New frequency
        frequency = self.frequencies[arm_index] + 1
        # New expected Value
        expected_value = sum / frequency

        #  log
        self.sums[arm_index] = sum
        self.frequencies[arm_index] = frequency


        delta = reward + self.discount * max(self.q_values) - self.q_values[arm_index]
        self.q_values[arm_index] = self.q_values[arm_index] + self.alpha * delta

        # Update epsilon
        v = math.e ** (-abs(self.alpha * delta) / self.sigma)
        f = (1 - v) / (1 + v)
        self.epsilon = self.d * f + (1 - self.d) * self.epsilon

        return


# configuration
arms = [
    'Configuration a',
    'Configuration b',
    'Configuration c',
    'Configuration d',
    'Configuration e',
    'Configuration f'
]

# instantiate bandits
bandit = Bandit(arms)
ref_bandit = reference_bandit.ReferenceBandit(arms)
