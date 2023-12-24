import os
from joblib import Memory
import copy
import getpass

user_name = getpass.getuser()

_disk_memory = Memory(location=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"), verbose=0)
def disk_cache(func, *args, **kwargs):
    return _disk_memory.cache(func, *args, **kwargs)

_tmpfs_memory = Memory(location=f'/tmp/{user_name}-cachedir', verbose=0)
def tmpfs_cache(func, *args, **kwargs):
    return _tmpfs_memory.cache(func, *args, **kwargs)

def ram_cache(func):
    ram_mem = {}
    def wrapper(*args, **kwargs):
        key = args, frozenset(kwargs.items())
        if key not in ram_mem:
            ram_mem[key] = func(*args, **kwargs)
        return copy.deepcopy(ram_mem[key]) # be careful with in-place op
    return wrapper

if __name__ == "__main__":
    # https://stackoverflow.com/questions/39020217/how-to-use-joblib-memory-of-cache-the-output-of-a-member-function-of-a-python-cl
    class A:
        def __init__(self):
            self.run = tmpfs_cache(self.run, ignore=['self'])
        def run(self, a, b):
            import time
            time.sleep(1)
            return a + b
    class B:
        def __init__(self):
            self.run = tmpfs_cache(self.run, ignore=['self'])
        def run(self, a, b):
            import time
            time.sleep(1)
            return a - b
        
    a = A()
    b = B()

    print(a.run(1, 2))
    print(a.run(1, 2))
    print(b.run(1, 2))
    print(b.run(1, 2))