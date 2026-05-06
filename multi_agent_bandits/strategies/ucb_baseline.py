import math
import random
from multi_agent_bandits.core.agent import Agent

class UCB_BaselineAgent(Agent):
    """
    Baseline UCB implementation.
    Chooses arm based on: estimated_value + exploration_bonus
    where exploration_bonus shrinks with more pulls
    """

    def __init__(self, n_arms, name = None):
        super().__init__(n_arms, name=name)

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_steps = 0
        self.last_arm = None

    def choose_arm(self):
        self.total_steps += 1

        #ensure we try each arm at least once, in random order to break agent symmetry
        unpulled = [arm for arm in range(self.n_arms) if self.counts[arm] == 0]
        if unpulled:
            self.last_arm = random.choice(unpulled)
            return self.last_arm


        ucb_scores = []
        for arm in range(self.n_arms):
            bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
            ucb_scores.append(self.values[arm] + bonus)

        #pick best, shuffle first so ties break randomly
        indices = list(range(self.n_arms))
        random.shuffle(indices)
        self.last_arm = max(indices, key=lambda a: ucb_scores[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])
