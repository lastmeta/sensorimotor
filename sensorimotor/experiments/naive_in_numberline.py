def explore():
    from sensorimotor.agents import NaiveSensorimotor
    from sensorimotor.envs import NumberLine
    env = NumberLine()
    env.seed(0)
    agent = NaiveSensorimotor(env)
    for i_episode in range(1):
        obs = env.reset()
        env.render()
        for t_timesteps in range(1000):
            action = agent.random_step(obs)
            obs, reward, done, info = env.step(action)
    env.close()
    return agent


if __name__ == '__main__':
    agent = explore()
    print(agent.get_path(target=11, start=0))
    print(agent.get_path(target=agent.previous.name, start=0))
    print(agent.previous.name)
