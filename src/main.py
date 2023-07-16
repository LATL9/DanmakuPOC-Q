from common import *

from game import *
from gui import *

from pyray import *
import threading

def train():
    rewards = {}
    threads = []
    
    for i in range(0, NUM_MODELS, round(NUM_MODELS / NUM_PROCESSES)):
        threads.append(ThreadWithResult(target=test, args=(list(range(i, i + round(NUM_MODELS / NUM_PROCESSES))),)))
    for i in range(len(threads)): threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        rewards = {**rewards, **threads[i].result}
    return rewards

def test(indexes):
    r = {}
    for i in range(round(NUM_MODELS / NUM_PROCESSES)):
        g = Game()
        for j in range(60 * 30): g.Update([bool(random.getrandbits(1)) for k in range(4)])
        r[indexes[i]] = g.End()
    return r

if __name__ == '__main__':
    gui = ThreadWithResult(target=gui)
    gui.start()

    rewards = train()
    print(rewards)

    # De-Initialization
    close_window()