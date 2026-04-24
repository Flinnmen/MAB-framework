from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm

from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.social_ucb import Social_UCB
from multi_agent_bandits.strategies.social_epsilon_greedy import Social_EpsilonGreedyAgent
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent

from multi_agent_bandits.core.reward_sharing import *


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):

    n_agents = 4

    arms = [
    Arm(mean=3.0, sd=1.0),   # clearly best
    Arm(mean=2.7, sd=1.5),
    Arm(mean=2.5, sd=1.0),

    Arm(mean=2.0, sd=1.0),   # mid cluster
    Arm(mean=2.0, sd=1.0),
    Arm(mean=1.8, sd=1.0),

    Arm(mean=1.5, sd=1.2),   # noisy mid-low group
    Arm(mean=1.5, sd=1.2),
    Arm(mean=1.2, sd=1.0),

    Arm(mean=1.0, sd=1.0),   # low group
    Arm(mean=1.0, sd=1.0),
    Arm(mean=0.8, sd=1.0),

    Arm(mean=0.5, sd=1.5)    # trap (high variance, low mean)
    ]

    env = Environment(
        n_agents=n_agents,
        arms=arms,
        collision_policy=raw
    )

    agents = [
        #Strategy here means trust strategy
        Social_EpsilonGreedyAgent(env.n_arms, env.n_agents, truthful=True, strategy='e_greedy'),
        Social_EpsilonGreedyAgent(env.n_arms, env.n_agents, truthful=True, strategy='ucb'),
        Social_UCB(env.n_arms, env.n_agents, truthful=True, strategy='e_greedy'),
        Social_UCB(env.n_arms, env.n_agents, truthful=True, strategy='ucb'),
    ]
    '''
    agents = [
        #Strategy here means trust strategy
        EpsilonGreedyAgent(env.n_arms),
        EpsilonGreedyAgent(env.n_arms),
        UCB_BaselineAgent(env.n_arms),
        UCB_BaselineAgent(env.n_arms),
    ]'''

    runner = ExperimentRunner(
        env,
        agents,
        timestep_limit=steps,
        save_dir=save_dir
    )

    runner.run(
        plot_rewards=plot_rewards,
        plot_frequencies=plot_frequencies
    )

    runner.print_summary()
