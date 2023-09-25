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
    torch.set_num_threads(NUM_PROCESSES)
    model = nn.Sequential(
        nn.ConstantPad2d(7, 1),
        nn.Conv2d(2, 16, kernel_size=(15, 15)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.ConstantPad2d(3, 1),
        nn.Conv2d(16, 64, kernel_size=(7, 7)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.ConstantPad2d(1, 1),
        nn.Conv2d(64, 256, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Flatten(1, 3),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 4 * FRAMES_PER_ACTION),
        nn.Sigmoid(),
        nn.ReLU()
    ).to(device)

    # optimisation
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    try:
        checkpoint = torch.load("models/model.pt", map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
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
        log.write("Time, Epoch, Fitness, Error\n") # header
        rankings = open("rankings.csv", 'w')
        rankings.write("Epoch, Seed\n") # header

    while True:
        epoch += 1

        if TRAIN_MODEL:
            seed = int(time.time())

        error = 0.0
        results = train()
        fitness = results['fitness']

        if not TRAIN_MODEL:
            print("Epoch {}: Fitness = {}, Q-Learning Fitness = {}, Error = {}".format(
                epoch,
                results['q_fitness'],
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

        print("Epoch {}: Error = {}".format(
            epoch,
            error
        ))
