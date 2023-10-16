from common import *

from game import *
from model import *

from pyray import *
import os
import time

#from torch.utils.tensorboard import SummaryWriter
#
#writer = SummaryWriter()

def train():
    return test(device, seed, model)

def test(device, seed, _model): # _ prevents naming conflict
    m = NNModel(device, seed, _model)
    fitness = m.train()
    return fitness

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(NUM_PROCESSES)
    model = nn.Sequential(
        nn.ConstantPad2d(7, 1),
        nn.Conv2d(2, 16, kernel_size=(15, 15)),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.ConstantPad2d(3, 1),
        nn.Conv2d(16, 64, kernel_size=(7, 7)),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.ConstantPad2d(1, 1),
        nn.Conv2d(64, 256, kernel_size=(3, 3)),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Flatten(1, 3),
        nn.Linear(4096, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 4 * FRAMES_PER_ACTION),
        nn.Sigmoid(),
        nn.ReLU()
    ).to(device)

    # optimisation
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    try:
        training_loader = torch.load("training_loaders/training_loader.pt", map_location=device)

    except FileNotFoundError:
        exit("no training loader found. exiting.")

    try:
        checkpoint = torch.load("models/model.pt", map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if not TRAIN_MODEL:
            log_r = open("log.csv", 'r').readlines()
            seed = int(log_r[len(log_r) - 1].split(', ')[1])
        print("Restarting from checkpoint at epoch {}.".format(epoch))
        log = open("log.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log = open("log.csv", 'w')
        log.write("Time, Seed, Epoch, Error, Fitness, Hits, Grazes, Nears\n") # header

    # TODO: figure out how to calculate number of batches given a DataLoader
    for i, (inputs, targets) in enumerate(training_loader):
        size = i
    size += 1

    while True:
        epoch += 1

        if TRAIN_MODEL:
            seed = int(time.time())

        error = 0.0
        results = train()
        fitness = results['fitness']

        if not TRAIN_MODEL:
            print("Epoch {}: Error = {}, Fitness = {}, Hits = {}, Grazes = {}, Nears = {}".format(
                epoch,
                error,
                fitness,
                results['hits'],
                results['grazes'],
                results['nears']
            ))
            exit() # if testing, exit now

        for i, (inputs, targets) in enumerate(training_loader):
            optimizer.zero_grad()
            y = model(inputs)
            loss = criterion(y, targets)
            error += float(loss)
            loss.backward()
            #for name, param in model.named_parameters():
            #    writer.add_histogram(name + '/grad', param.grad, global_step=epoch)
            optimizer.step()
            print("Batch {}".format(i), end='\r')
        error /= size + 1

        log.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(
            time.asctime(),
            seed, # used for replays
            epoch,
            error, 
            fitness,
            results['hits'],
            results['grazes'],
            results['nears']
        ))
        log.flush()

        if True:
            checkpoint = {'epoch': epoch}
            checkpoint['model_state_dict'] = model.state_dict()
            torch.save(checkpoint, "models/model-{}.pt".format(epoch))
            os.system("cp models/model-{}.pt models/model.pt".format(epoch)) # model.pt = most recent

        print("Epoch {}: Error = {}, Fitness = {}, Hits = {}, Grazes = {}, Nears = {}".format(
            epoch,
            error,
            fitness,
            results['hits'],
            results['grazes'],
            results['nears']
        ))
