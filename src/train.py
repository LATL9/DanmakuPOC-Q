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

def main():
    device = torch.device('cpu')
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
        nn.Conv2d(64, 128, kernel_size=(3, 3)),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Flatten(1, 3),
        nn.Linear(2048, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 64),
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
        # use first 10% of training dataset as validation dataset
        validation_loader = torch.utils.data.DataLoader(
            QDataset(
                inps=training_loader.dataset.inps[:len(training_loader.dataset.inps) // 10],
                outs=training_loader.dataset.outs[:len(training_loader.dataset.outs) // 10]
            ),
            batch_size=16,
            shuffle=True,
            num_workers=1
        )
        # remove first 10% from training dataset
        training_loader.dataset.inps = training_loader.dataset.inps[len(training_loader.dataset.inps) // 10:]
        training_loader.dataset.outs = training_loader.dataset.outs[len(training_loader.dataset.outs) // 10:]

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
        log.write("Time, Seed, Epoch, Training Error, Validation Error, Fitness, Hits, Grazes, Nears\n") # header

    # TODO: figure out how to calculate number of batches given a DataLoader
    for i, (inputs, targets) in enumerate(training_loader):
        training_loader_size = i
    training_loader_size += 1
    for i, (inputs, targets) in enumerate(validation_loader):
        validation_loader_size = i
    validation_loader_size += 1

    while True:
        epoch += 1

        if TRAIN_MODEL:
            seed = int(time.time())

        training_error = 0.0
        validation_error = 0.0
        results = train()
        fitness = results['fitness']

        if not TRAIN_MODEL:
            print("Epoch {}: Fitness = {}, Hits = {}, Grazes = {}, Nears = {}".format(
                epoch,
                fitness,
                results['hits'],
                results['grazes'],
                results['nears']
            ))
            exit() # if testing, exit now

        # training
        for i, (inputs, targets) in enumerate(training_loader):
            optimizer.zero_grad()
            y = model(inputs)
            loss = criterion(y, targets)
            training_error += float(loss)
            loss.backward()
            optimizer.step()
            print("Batch {}".format(i), end='\r')
        # validation (no training)
        for i, (inputs, targets) in enumerate(validation_loader):
            y = model(inputs)
            loss = criterion(y, targets)
            validation_error += float(loss)
            print("Batch {}".format(i), end='\r')

        training_error /= training_loader_size
        validation_error /= validation_loader_size

        log.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
            time.asctime(),
            seed, # used for replays
            epoch,
            training_error,
            validation_error,
            fitness,
            results['hits'],
            results['grazes'],
            results['nears']
        ))
        log.flush()

        if not epoch % 10:
            checkpoint = {'epoch': epoch}
            checkpoint['model_state_dict'] = model.state_dict()
            torch.save(checkpoint, "models/model-{}.pt".format(epoch))
            os.system("cp models/model-{}.pt models/model.pt".format(epoch)) # model.pt = most recent

        print("Epoch {}: Training Error = {}, Validation Error = {}, Fitness = {}, Hits = {}, Grazes = {}, Nears = {}".format(
            epoch,
            training_error,
            validation_error,
            fitness,
            results['hits'],
            results['grazes'],
            results['nears']
        ))
