from common import *

from game import *
from model import *

from pyray import *
import os
import time

def train():
    jobs = []
    manager = mp.Manager()
    fitnesses = manager.dict()
   
    for i in range(NUM_PROCESSES):
        jobs.append(mp.Process(target=test, args=(fitnesses, device, seed, list(range(i * NUM_MODELS_PER_PROCESS, (i + 1) * NUM_MODELS_PER_PROCESS)), models[i],)))
    if TEST_MODEL == -1:
        for i in range(len(jobs)): 
            jobs[i].start()
        for i in range(len(jobs)):
            jobs[i].join()
    else:
        # if testing, only test thread containing specified model
        jobs[TEST_MODEL // NUM_MODELS_PER_PROCESS].start()
        jobs[TEST_MODEL // NUM_MODELS_PER_PROCESS].join()

    return fitnesses

def test(fitnesses, device, seed, indexes, _models): # _ prevents naming conflict
    for i in range(NUM_MODELS_PER_PROCESS):
        m = NNModel(device, seed, indexes[i], _models[i])
        fitnesses[indexes[i]] = m.train()
        print("{} / {}".format(len(fitnesses), NUM_MODELS), end='\r')

if __name__ == '__main__':
    device = torch.device("cpu")
    models = [
        [nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(9, 9), padding=4, bias=False),
            nn.MaxPool2d((4, 4), stride=4),
            nn.Conv2d(4, 8, kernel_size=(5, 5), padding=2, bias=False),
            nn.MaxPool2d((4, 4), stride=4),
            nn.Flatten(0, -1),
            nn.Linear(32, 4, bias=False)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]

    try:
        checkpoint = torch.load("models/models.pt", map_location=device)
        epoch = checkpoint['epoch']
        for i in range(NUM_MODELS):
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].load_state_dict(checkpoint['model{}_state_dict'.format(i)])
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].to(device)

        print("Restarting from checkpoint at epoch {}.".format(epoch))
        log = open("log.csv", 'a')
        rankings = open("rankings.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log = open("log.csv", 'w')
        log.write("Time, Epoch, Median, 1st Quartile Avg, 3rd Quartile Avg\n") # header
        rankings = open("rankings.csv", 'w')
        rankings.write("Epoch, Models\n") # header

    stats = ""
    while True:
        epoch += 1

        seed = time.time()
        fitnesses = train()
        fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}

        if TEST_MODEL != -1: exit() # if testing, exit now

        # mutation (1st quartile = highest fitness, 4th quarter = lowest fitness):
        # 4th quartile is replaced with new random weights
        # 3rd quartile is replaced by 1st quarter with mutations
        # 2nd quartile is mutated
        # 1st quartile stays same
        vals = list(fitnesses.values())
        
        median = vals[((NUM_MODELS * 3 // 4) - 1) // 2]
        quartile_1_avg = sum(vals[:NUM_MODELS // 4]) // (NUM_MODELS // 4)
        quartile_3_avg = sum(vals[NUM_MODELS // 2:NUM_MODELS * 3 // 4]) // (NUM_MODELS // 4)

        # 2nd quartile
        for i in range(NUM_MODELS // 4, NUM_MODELS // 2):
            # mutation
            for param in models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER / 4 * torch.randn_like(param).to(device)

        # 3rd quartile
        for i in range(NUM_MODELS // 4):
            models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS]

            for param in models[(i + NUM_MODELS // 2) // NUM_MODELS_PER_PROCESS][(i + NUM_MODELS // 2) % NUM_MODELS_PER_PROCESS].parameters():
                param.data += MUTATION_POWER * torch.randn_like(param).to(device)

        # 4th quartile
        for i in range(NUM_MODELS * 3 // 4, NUM_MODELS):
            for param in models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].parameters():
                param.data = torch.randn_like(param).to(device)

        log.write("{}, {}, {}, {}, {}\n".format(
            time.asctime(),
            epoch,
            median,
            quartile_1_avg,
            quartile_3_avg
        ))
        log.flush()
        # 1st column is epoch, subsequent even columns = model index and odd columns = model fitness ordered in descending order of fitness
        rankings.write("{}, {}\n".format(epoch, ", ".join(["{}, {}".format(x, fitnesses[x]) for x in fitnesses])))
        rankings.flush()

        #if epoch % 5 == 0:
        if True:
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
