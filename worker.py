from multiprocessing import Process

class Worker(Process):
    def __init__(self):
        super().__init__()

    def run(self):
        pass