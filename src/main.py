from common import *

from game import *
from gui import *

from pyray import *
#import threading
#import torch
#import torch.nn as nn
#import torch.nn.functional as F

def train():
    rewards = {}
    threads = []
    
    for i in range(0, NUM_MODELS, round(NUM_MODELS / NUM_PROCESSES)):
        threads.append(ThreadWithResult(target=test, args=(device, list(range(i, i + round(NUM_MODELS / NUM_PROCESSES))),)))
    for i in range(len(threads)): threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        rewards = {**rewards, **threads[i].result}
    return rewards

def test(device, indexes):
    r = {}
    for i in range(round(NUM_MODELS / NUM_PROCESSES)):
        g = Game(device)
        for j in range(60 * 30): g.Update([bool(random.getrandbits(1)) for k in range(4)])
        r[indexes[i]] = g.End()
    return r

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gui = ThreadWithResult(target=gui, args=(device,))
    gui.start()

    rewards = train()
    print(rewards)

    # De-Initialization
    close_window()