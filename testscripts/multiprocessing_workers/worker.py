from multiprocessing import Process

class Worker(Process):
    """ Base Class for Agent Processes to ensure multiprocessing
    """
    def __init__(self):
        super().__init__()

    def run(self):
        pass