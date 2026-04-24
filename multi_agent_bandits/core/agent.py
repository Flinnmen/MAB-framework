import math
import random

class Agent:
    """
    Base interface for all agents.
    """

    def __init__(self, n_arms, social_epsilon=0.1, name=None):
        self.n_arms = n_arms
        self.name = name if name is not None else self.__class__.__name__
        self.agent_steps = 0
        self.social_epsilon = social_epsilon

    def ucb(self):
        self.agent_steps += 1
        #Check untried agents.
        for agent in range(self.n_agents):
            if self.agentCounts[agent] == 0:
                self.last_agent = agent
                return agent
        
        #Exploit best agents.
        agent_scores = []
        for agent in range(self.n_agents):
            bonus = math.sqrt((2 * math.log(self.agent_steps + 1)) / self.agentCounts[agent])
            agent_scores.append(self.trusts[agent] + bonus)
        self.last_agent = max(range(self.n_agents), key=lambda a: agent_scores[a])
        return self.last_agent
    
    def e_greedy(self):
        if random.random() < self.social_epsilon:
            self.last_agent = random.randrange(self.n_agents)
            return self.last_agent

        self.last_agent = max(range(self.n_agents), key=lambda a: self.trusts[a])
        return self.last_agent

    def choose_arm(self):
        raise NotImplementedError

    def update(self, reward):
        pass
