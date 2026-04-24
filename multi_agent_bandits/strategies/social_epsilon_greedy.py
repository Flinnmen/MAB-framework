import random
from multi_agent_bandits.core.agent import Agent

class Social_EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms, n_agents, truthful, strategy, epsilon=0.1, name = None):
        super().__init__(n_arms, name=name)
        self.epsilon = epsilon

        #track estimates and counts
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.last_arm = None

        self.n_agents = n_agents
        self.trusts = [0.5] * n_agents
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
        if random.random() < self.epsilon:
            self.last_arm = random.randrange(self.n_arms)
            return self.last_arm

        
        boosted_scores = self.values.copy()

        #Doubled score for arm that is boosed
        if self.boosted_arm is not None:
            trust = self.trusts[self.last_agent]
            boosted_scores[self.boosted_arm] += 0.5 * trust

        self.last_arm = max(range(self.n_arms), key=lambda a: boosted_scores[a])
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
