# use click
# specify environment and agent
# tell it to explore
# once it is done, you can inspect it's mind
# you can reset it's location in env
# you can tell it to move to new location


def demo():
    ''' demo mode is highly limited, made for demo '''
    from sensorimotor.lib import MetaEnvironment
    MetaEnvironment()


if __name__ == '__main__':
    demo()
