def linear_share(raw_reward, n_agents):
    """Equal split (default in the environment)."""
    return [raw_reward / n_agents] * n_agents

def zero_on_collision(raw_reward, n_agents):
    """Everyone gets zero in case of collision."""
    return [0.0] * n_agents

def winner_takes_all(raw_reward, n_agents):
    """Pick one agent randomly to get the whole reward"""
    import random
    rewards = [0.0] * n_agents
    winner = random.randrange(n_agents)
    rewards[winner] = raw_reward
    return rewards

def raw(raw_reward, n_agents):
    return [raw_reward] * n_agents

