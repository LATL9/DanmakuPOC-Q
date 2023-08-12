from common import *

from game import *
from model import *

from pyray import *
import time
import threading

def train():
    fitnesses = {}
    threads = []
   
    conv1s = [
        [nn.Sequential(
            nn.Conv2d(1, 1, 16, 1, 2),
            nn.MaxPool2d(2)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]
    conv2s = [
        [nn.Sequential(
            nn.Conv2d(1, 1, 4, 1, 2),
            nn.MaxPool2d(3)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]
    outs = [
        [nn.Sequential(
            nn.Linear(4, 1)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]

    for i in range(NUM_PROCESSES):
        threads.append(ThreadWithResult(target=test, args=(device, list(range(i * NUM_MODELS_PER_PROCESS, (i + 1) * NUM_MODELS_PER_PROCESS)), conv1s[i], conv2s[i], outs[i],)))
    for i in range(len(threads)): 
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        fitnesses = {**fitnesses, **threads[i].result}
    return fitnesses

def test(device, indexes, conv1s, conv2s, outs):
    r = {}
    for i in range(NUM_MODELS_PER_PROCESS):
        m = NNModel(device, indexes[i], conv1s[i], conv2s[i], outs[i])
        r[indexes[i]] = m.train()
    return r

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 1

    fitnesses = train()
    fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}
    vals = list(fitnesses.values())
    print("Epoch {}: Mean = {}, Median = {}, Top 50% = {}, Bottom 50% = {}".format(
        epoch,
        sum(fitnesses.values()) // len(fitnesses),
        vals[(len(fitnesses) + 1) // 2],
        sum(vals[:len(fitnesses) // 2]) // (len(fitnesses) // 2),
        sum(vals[len(fitnesses) // 2:]) // (len(fitnesses) // 2)
    ))
