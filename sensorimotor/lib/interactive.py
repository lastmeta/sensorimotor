class MetaEnvironment():
    def __init__(self):
        self.loop()

    def loop(self):
        command = input('command?')
        while command != 'exit':
            if command == 'help':
                self.help()
            elif command.split()[0] == 'init':
                self.init(command.split()[1], command.split()[2])
            elif command.split()[0] == 'explore':
                self.explore(int(command.split()[1]))
            elif command.split()[0] == 'seen':
                self.seen(command.split()[1])
            elif command.split()[0] == 'plan':
                self.plan(command.split()[1])
            elif command.split()[0] == 'make':
                self.make(command.split()[1])
            elif command.split()[0] == 'reset':
                self.reset(command.split()[1])

    def help(self):
        print('commands:')
        print('exit, help, init <agent> <env>, explore <steps>, seen <state>, plan <state>, make <state>, reset <state>')
        print('agents: NaiveSensorimotor, SensorimotorAutoencoder, FullyConnectedSensorimotorAutoencoderAgents, SensorimotorAutoencoderAgents')
        print('environments: NumberLine, SimpleCube, RubiksCube')

    def init(self, agent, env):
        from sensorimotor.agents import NaiveSensorimotor
        from sensorimotor.envs import NumberLine
        self.env = NumberLine()
        self.env.seed(0)
        self.agent = NaiveSensorimotor(self.env)
        for i_episode in range(1):
            obs = self.env.reset()
            self.env.render()

    def close(self, env):
        self.env.close()

    def explore(self, steps):
        for t_timesteps in range(steps):
            action = self.agent.random_step(obs)
            obs, reward, done, info = self.env.step(action)

    def plan(self, state):
        print(agent.get_path(target=state))

    def make(self, state):
        ''' have not implmented agent.do '''
        agent.do(agent.get_path(target=state))

    def reset(self, state):
        ''' set env to state pattern on self.agent and self.env '''

    def seen(self, state):
        ''' has this agent seen this state before? '''


if __name__ == '__main__':
    print('use the command line interface by typing sensorimotor demo')
