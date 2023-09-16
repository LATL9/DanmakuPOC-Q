from common import *

from game import *
from model import *

from pyray import *
import os
import time
import torch.multiprocessing as mp

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
        nn.Conv2d(2, 4, kernel_size=(9, 9)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.ConstantPad2d(2, 1),
        nn.Conv2d(4, 8, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Flatten(1, 3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 4 * FRAMES_PER_ACTION),
        nn.Sigmoid(),
        nn.ReLU(),
     ).to(device)

    # optimisation
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

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
        log.write("Time, Epoch, FItness, Error\n") # header
        rankings = open("rankings.csv", 'w')
        rankings.write("Epoch, Seed\n") # header

    while True:
        epoch += 1

        if TRAIN_MODEL:
            seed = int(time.time()) # random seed
            error = 0.0

        results = train()
        fitness = results['fitness']

        if not TRAIN_MODEL:
            print("Epoch {}: Fitness = {}, Error = {}".format(
                epoch,
                fitness,
                error
            ))
            exit() # if testing, exit now

        training_data = QDataset(
            inps=results['exp_inps'],
            outs=results['exp_outs']
        )
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=16,
            shuffle=True,
            num_workers=1
        )

        for i, (inputs, targets) in enumerate(training_loader):
            optimizer.zero_grad()
            y = model(inputs)
            loss = criterion(y, targets)
            error += float(loss)
            loss.backward()
            optimizer.step()

        log.write("{}, {}, {}, {}\n".format(
            time.asctime(),
            epoch,
            fitness,
            error 
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

        print("Epoch {}: Fitness = {}, Error = {}".format(
            epoch,
            fitness,
            error
        ))
