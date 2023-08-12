from common import *

from game import *
from model import *

from pyray import *
import time
import threading

def train():
    fitnesses = {}
    threads = []
   
    for i in range(NUM_PROCESSES):
        threads.append(ThreadWithResult(target=test, args=(device, list(range(i * NUM_MODELS_PER_PROCESS, (i + 1) * NUM_MODELS_PER_PROCESS)), models[i],)))
    for i in range(len(threads)): 
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        fitnesses = {**fitnesses, **threads[i].result}
    return fitnesses

def test(device, indexes, _models): # _ presents naming conflict
    r = {}
    for i in range(NUM_MODELS_PER_PROCESS):
        m = NNModel(device, indexes[i], _models[i])
        r[indexes[i]] = m.train()
    return r

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = [
        [nn.Sequential(
            nn.Conv2d(1, 1, 16, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(1, 1, 4, 1, 2),
            nn.MaxPool2d(3),
            nn.Linear(4, 1)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]

    epoch = 0
    while True:
        epoch += 1

        fitnesses = train()
        fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}
        vals = list(fitnesses.values())

        # mutation (1st quarter = highest fitness, 4th quarter = lowest fitness):
        # 4th quarter is replaced with new random weights
        # 3rd quarter is replaced by 1st quarter with mutations
        # 2nd quarter is mutated
        # 1st quarter stays same
        keys = list(fitnesses.keys())
        
        # 2nd quarter
        for i in range(NUM_MODELS // 4, NUM_MODELS // 2):
            for param in models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param)

        # 3rd quarter
        for i in range(NUM_MODELS // 4):
            models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS]

            # mutation
            for param in models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param)

        # 4th quarter
        for i in range(NUM_MODELS * 3 // 4, NUM_MODELS):
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS] = nn.Sequential(
                nn.Conv2d(1, 1, 16, 1, 2),
                nn.MaxPool2d(2),
                nn.Conv2d(1, 1, 4, 1, 2),
                nn.MaxPool2d(3),
                nn.Linear(4, 1)
            ).to(device)

        print("Epoch {}: Mean = {}, Median = {}, Top 50% = {}, Bottom 50% = {}".format(
            epoch,
            sum(fitnesses.values()) // NUM_MODELS,
            vals[(NUM_MODELS + 1) // 2],
            sum(vals[:NUM_MODELS // 2]) // (NUM_MODELS // 2),
            sum(vals[NUM_MODELS // 2:]) // (NUM_MODELS // 2)
        ))
