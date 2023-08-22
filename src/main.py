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

    if TEST_MODEL == -1:
        jobs = [mp.Process(target=test, args=(fitnesses, device, seed, list(range(i * NUM_MODELS_PER_PROCESS, (i + 1) * NUM_MODELS_PER_PROCESS)), models[i],)) for i in range(NUM_PROCESSES)]
        for i in range(len(jobs)): jobs[i].start()
        for i in range(len(jobs)): jobs[i].join()
    else:
        # if testing, only test specified model
        test(fitnesses, device, seed, [TEST_MODEL], [models[TEST_MODEL // NUM_MODELS_PER_PROCESS][TEST_MODEL % NUM_MODELS_PER_PROCESS]])

    return fitnesses

def test(fitnesses, device, seed, indexes, _models): # _ prevents naming conflict
    for j in range(len(indexes)):
        m = NNModel(device, seed, indexes[j], _models[j])
        fitnesses[indexes[j]] = m.train()
        print("{} / {}".format(len(fitnesses), NUM_MODELS), end='\r')

if __name__ == '__main__':
    device = torch.device("cpu")
    torch.set_num_threads(1)
    models = [
        [nn.Sequential(
            nn.ConstantPad2d(4, 1),
            nn.Conv2d(2, 4, kernel_size=(9, 9), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ConstantPad2d(2, 1),
            nn.Conv2d(4, 8, kernel_size=(5, 5), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Flatten(0, -1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        ).to(device) for i in range(NUM_MODELS_PER_PROCESS)]
        for j in range(NUM_PROCESSES)
    ]

    try:
        checkpoint = torch.load("models/models.pt", map_location=device)
        epoch = checkpoint['epoch']
        for i in range(NUM_MODELS):
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].load_state_dict(checkpoint['model{}_state_dict'.format(i)])
            models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].to(device)

        if TEST_MODEL != -1:
            rankings_r = open("rankings.csv", 'r').readlines()
            seed = float(rankings_r[len(rankings_r) - 1].split(', ')[1])
        print("Restarting from checkpoint at epoch {}.".format(epoch))
        log = open("log.csv", 'a')
        rankings = open("rankings.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log = open("log.csv", 'w')
        log.write("Time, Epoch, Best, Median, 1st Quartile Avg, 3rd Quartile Avg\n") # header
        rankings = open("rankings.csv", 'w')
        rankings.write("Epoch, Seed, Models\n") # header

    while True:
        epoch += 1

        if TEST_MODEL == -1: seed = int(time.time())
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

        log.write("{}, {}, {}, {}, {}, {}\n".format(
            time.asctime(),
            epoch,
            vals[0],
            median,
            quartile_1_avg,
            quartile_3_avg
        ))
        log.flush()
        # 1st column is epoch, 2nd column is seed (for replays), subsequent odd columns = model index and even columns = model fitness ordered in descending order of fitness
        rankings.write("{}, {}, {}\n".format(
            epoch,
            seed,
            ", ".join(["{}, {}".format(x, fitnesses[x]) for x in fitnesses])
        ))
        rankings.flush()

        #if epoch % 5 == 0:
        if True:
            fitnesses = {k: v for k, v in sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)}
            checkpoint = {'epoch': epoch}
            for i in range(NUM_MODELS):
                checkpoint['model{}_state_dict'.format(i)] = models[i // NUM_MODELS_PER_PROCESS][i % NUM_MODELS_PER_PROCESS].state_dict()
            torch.save(checkpoint, "models/models-{}.pt".format(epoch))
            os.system("cp models/models-{}.pt models/models.pt".format(epoch)) # models.pt = most recent

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

        print("Epoch {}: Median = {}, Best = {}, 1st Quartile Avg = {}, 3rd Quartile Avg = {}\n".format(
            epoch,
            median,
            vals[0],
            quartile_1_avg,
            quartile_3_avg
        ), end='')
