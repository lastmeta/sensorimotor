import time


def time_it(func, *args, **kwargs):
    then = time.time()
    result = func(*args, **kwargs)
    print('seconds:', time.time() - then)
    return result
