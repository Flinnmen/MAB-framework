from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm

from multi_agent_bandits.strategies.truth_ucb import UCB_Truth
from multi_agent_bandits.strategies.liar_ucb import UCB_Lie
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):

    n_agents = 3

    arms = [
        Arm(mean=1.0, sd=1.0),
        Arm(mean=2.0, sd=1.0),
        Arm(mean=1.5, sd=1.0)
    ]

    env = Environment(
        n_agents=n_agents,
        arms=arms
    )

    agents = [
        UCB_Lie(env.n_arms, env.n_agents),
        UCB_Truth(env.n_arms, env.n_agents),
        UCB_Truth(env.n_arms, env.n_agents)
    ]

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
