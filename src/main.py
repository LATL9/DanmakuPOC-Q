from common import *

from game import *
from model import *

from pyray import *
import os
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

    try:
        checkpoint = torch.load("models/models.pt")
        epoch = checkpoint['epoch']
        for i in range(NUM_MODELS):
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].load_state_dict(checkpoint['model{}_state_dict'.format(i)])

        print("Restarting from checkpoint at epoch {}.".format(epoch))
        log = open("log.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log = open("log.csv", 'w')
        log.write("Epoch, Median, 1st Quartile Avg, 3rd Quartile Avg\n") # header

    stats = ""
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
        
        median = vals[((NUM_MODELS * 3 // 4) - 1) // 2]
        quartile_1_avg = sum(vals[:NUM_MODELS // 4]) // (NUM_MODELS // 4)
        quartile_3_avg = sum(vals[NUM_MODELS * 3 // 4:]) // (NUM_MODELS // 4)

        # 2nd quartile
        for i in range(NUM_MODELS // 4, NUM_MODELS // 2):
            # mutation
            for param in models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param).to(device)

        # 3rd quartile
        for i in range(NUM_MODELS // 4):
            models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS]

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

        log.write("{}, {}, {}, {}\n".format(
            epoch,
            median,
            quartile_1_avg,
            quartile_3_avg
        ))
        log.flush()

        if epoch % 5 == 0:
            fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}
            checkpoint = {'epoch': epoch}
            for i in range(NUM_MODELS):
                checkpoint['model{}_state_dict'.format(i)] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].state_dict()
            torch.save(checkpoint, "models/models-{}.pt".format(epoch))
            os.system("cp models/models-{}.pt models/models.pt".format(epoch)) # models.pt = most recent

        print("Epoch {}: Median = {}, 1st Quartile Avg = {}, 3rd Quartile Avg = {}\n".format(
            epoch,
            median,
            quartile_1_avg,
            quartile_3_avg
        ), end='')
