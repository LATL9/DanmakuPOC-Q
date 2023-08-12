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

    log = open("log.csv", 'w')
    log.write("Epoch, Median, 1st Quartile Avg, 3rd Quartile Avg\n") # header
    stats = ""
    epoch = 0
    while True:
        epoch += 1

        fitnesses = train()
        fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}
        vals = list(fitnesses.values())

        # mutation (1st quartile = highest fitness, 4th quarter = lowest fitness):
        # 4th quartile is replaced with new random weights
        # 3rd quartile is replaced by 1st quarter with mutations
        # 2nd quartile is mutated
        # 1st quartile stays same
        keys = list(fitnesses.keys())
        
        # 2nd quartile
        for i in range(NUM_MODELS // 4, NUM_MODELS // 2):
            for param in models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param).to(device)

        # 3rd quartile
        for i in range(NUM_MODELS // 4):
            models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS]

            # mutation
            for param in models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param).to(device)

        # 4th quartile
        for i in range(NUM_MODELS * 3 // 4, NUM_MODELS):
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS] = nn.Sequential(
                nn.Conv2d(1, 1, 16, 1, 2),
                nn.MaxPool2d(2),
                nn.Conv2d(1, 1, 4, 1, 2),
                nn.MaxPool2d(3),
                nn.Linear(4, 1)
            ).to(device)

        median = vals[(NUM_MODELS + 1) // 2]
        quartile_1_avg = sum(vals[:NUM_MODELS // 4]) // (NUM_MODELS // 4)
        quartile_3_avg = sum(vals[NUM_MODELS * 3 // 4:]) // (NUM_MODELS // 4)

        log.write("{}, {}, {}, {}\n".format(
            epoch,
            median,
            quartile_1_avg,
            quartile_3_avg
        ))
        log.flush()
        print("Epoch {}: Median = {}, 1st Quartile Avg = {}, 3rd Quartile Avg = {}\n".format(
            epoch,
            median,
            quartile_1_avg,
            quartile_3_avg
        ), end='')
