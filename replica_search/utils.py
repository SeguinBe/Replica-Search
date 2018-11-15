from time import time


class Timer:

    def __init__(self, description, disable=False):
        self.description = description
        self.disable = disable

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start
        if not self.disable:
            print("{} : {}".format(self.description, self.interval))
