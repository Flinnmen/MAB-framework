import math
import random
from multi_agent_bandits.core.agent import Agent

class UCB_Lie(Agent):
    """
    Baseline UCB implementation.
    Chooses arm based on: estimated_value + exploration_bonus
    where exploration_bonus shrinks with more pulls
    """

    def __init__(self, n_arms, n_agents, name = None):
        super().__init__(n_arms, name=name)

        self.n_agents = n_agents
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.agentCounts = [0] * n_agents
        self.trusts = [0.0] * n_agents
        self.total_steps = 0
        self.ucb_scores = []
        self.last_arm = None
        self.truthful = False
        self.last_agent = None
        self.boosted_arm = None
        self.askedAgents = {}
    
    def choose_agent(self):
        agent_scores = []
        for agent in range(self.n_agents):
            if self.agentCounts[agent] == 0:
                self.last_agent = agent
                return agent
            
        for agent in range(self.n_agents):
            bonus = math.sqrt((2 * math.log(self.total_steps + 1)) / (self.agentCounts[agent] + 1))
            agent_scores.append(self.trusts[agent] + bonus)
        return max(range(self.n_agents), key=lambda a: agent_scores[a])

    def choose_arm(self):
        self.total_steps += 1

        #ensure we try each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        for arm in range(self.n_arms):
            if arm == self.boosted_arm:
                bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
                self.ucb_scores.append((self.values[arm] + bonus)*2)
            else:
                bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
                self.ucb_scores.append(self.values[arm] + bonus)

        #pick best
        self.last_arm = max(range(self.n_arms), key=lambda a: self.ucb_scores[a])
        return self.last_arm

    def update(self, reward):
        agent = self.last_agent
        self.agentCounts[agent] += 1
        if self.askedAgents.get(self.boosted_arm) is None:
            self.askedAgents[self.boosted_arm] = {agent}
        else:
            self.askedAgents[self.boosted_arm].add(agent)
        

        arm = self.last_arm
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])
        if arm in self.askedAgents:
            for a in self.askedAgents[arm]:
                if reward > 2:
                    self.trusts[a] += 1
                else:
                    self.trusts[a] -= 1
            self.askedAgents.pop(arm)
