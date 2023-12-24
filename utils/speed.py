import time
import torch

ignore = set(["IGNORE"])
def dur(tag="NOTAG", sync=False):
    def decorator(func):
        if ("ALL" in ignore) or (tag in ignore):
            return func
        def wrapper(*args, **kwargs):
            if sync:
                torch.cuda.synchronize()
            start = time.time()
            result = func(*args, **kwargs)
            if sync:
                torch.cuda.synchronize()
            end = time.time()
            if ("ALL" not in ignore) and (tag not in ignore):
                print(f"{tag:<20} took {end-start:.2e} sec")
            return result
        return wrapper
    return decorator

class DurCtx:
    def __init__(self, tag="NOTAG", sync=False):
        self.tag = tag
        self.sync = sync
    def __enter__(self):
        if ("ALL" in ignore) or (self.tag in ignore):
            return
        if self.sync:
            torch.cuda.synchronize()
        self.start = time.time()
    def __exit__(self, exc_type, exc_value, traceback):
        if ("ALL" in ignore) or (self.tag in ignore):
            return
        if self.sync:
            torch.cuda.synchronize()
        end = time.time()
        print(f"{self.tag:<20} took {end-self.start:.2e} sec")

def quiet():
    ignore.add("ALL")

def slow_func(a):
    time.sleep(1)
    return 2 * a