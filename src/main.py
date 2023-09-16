from common import *

from game import *
from model import *

from pyray import *
import os
import time

def train():
    return test(device, seed, model)

def test(device, seed, _model): # _ prevents naming conflict
    m = NNModel(device, seed, _model)
    fitness = m.train()
    return fitness

if __name__ == '__main__':
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model = nn.Sequential(
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
        nn.Linear(32, 4 * FRAMES_PER_ACTION)
     ).to(device)

    try:
        checkpoint = torch.load("models/model.pt", map_location=device)
        epoch = checkpoint['epoch']
        for i in range(NUM_MODELS):
            model.load_state_dict(checkpoint['model_state_dict'.format(i)])
            model.to(device)

        if not TRAIN_MODEL:
            rankings_r = open("rankings.csv", 'r').readlines()
            seed = float(rankings_r[len(rankings_r) - 1].split(', ')[1])
        print("Restarting from checkpoint at epoch {}.".format(epoch))
        log = open("log.csv", 'a')
        rankings = open("rankings.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log = open("log.csv", 'w')
        log.write("Time, Epoch, Fitness\n") # header
        rankings = open("rankings.csv", 'w')
        rankings.write("Epoch, Seed\n") # header

    while True:
        epoch += 1

        if TRAIN_MODEL: seed = int(time.time())
        fitness = train()

        if not TRAIN_MODEL: exit() # if testing, exit now

        log.write("{}, {}, {}\n".format(
            time.asctime(),
            epoch,
            fitness
        ))
        log.flush()
        # 1st column is epoch, 2nd column is seed (for replays)
        rankings.write("{}, {}\n".format(
            epoch,
            seed
        ))
        rankings.flush()

        if True:
            checkpoint = {'epoch': epoch}
            checkpoint['model_state_dict'] = model.state_dict()
            torch.save(checkpoint, "models/model-{}.pt".format(epoch))
            os.system("cp models/model-{}.pt models/model.pt".format(epoch)) # model.pt = most recent

        print("Epoch {}: Fitness = {}\n".format(
            epoch,
            fitness,
        ), end='')
