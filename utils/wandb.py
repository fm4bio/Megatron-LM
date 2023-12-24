class DisableWandbConfig: # makes these ops ineffective
    def update(self, *args, **kwargs):
        print(f"update: {args}, {kwargs}")

class DisableWandb: # make these ops ineffective
    def __init__(self) -> None:
        self.config = DisableWandbConfig()

    def init(self, *args, **kwargs):
        print(f"init: {args}, {kwargs}")

    def log(self, *args, **kwargs):
        print(f"log: {args}, {kwargs}")