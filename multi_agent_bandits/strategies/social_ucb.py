import math
import random
from multi_agent_bandits.core.agent import Agent

class Social_UCB(Agent):
    """
    Baseline UCB implementation.
    Chooses arm based on: estimated_value + exploration_bonus
    where exploration_bonus shrinks with more pulls
    """

    def __init__(self, n_arms, n_agents, truthful, strategy, name = None):
        super().__init__(n_arms, name=name)

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_steps = 0
        self.last_arm = None

        self.n_agents = n_agents
        self.trusts = [0.0] * n_agents
        self.agentCounts = [0] * n_agents
        self.ucb_scores = []
        self.truthful = truthful
        self.last_agent = None
        self.boosted_arm = None
        self.askedAgents = {}
        if strategy == "ucb":
            self.agent_selection_strat = self.ucb
        elif strategy == "e_greedy":
            self.agent_selection_strat = self.e_greedy
        else:
            raise NotImplementedError
    
    def choose_agent(self):
        # Call the strategy
        self.last_agent = self.agent_selection_strat()
        return self.last_agent

    def choose_arm(self):
        self.total_steps += 1

        #ensure we try each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        self.ucb_scores = []
        for arm in range(self.n_arms):
            if arm == self.boosted_arm:
                trust = self.trusts[self.last_agent]
                bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
                self.ucb_scores.append((self.values[arm] + bonus) + 0.5*trust)
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

        #update the strategy with new info
        arm = self.last_arm
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm]) 

        #Updates correlated to trust system.
        average_performance = sum(self.values) / len(self.values)
        print(self.trusts)
        if arm in self.askedAgents:
            for a in self.askedAgents[arm]:
                #If the arm does worse than agents average then trust will decrease.
                learning_rate = 0.05
                if reward > average_performance:
                    self.trusts[a] += learning_rate
                elif reward < average_performance:
                    self.trusts[a] -= learning_rate
                
                #Keep it bounded.
                self.trusts[a] = max(0.0, min(self.trusts[a], 1.0))
                
            self.askedAgents.pop(arm)