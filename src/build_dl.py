from common import *

from game import *
from model import *

from pyray import *
import os
import time

if __name__ == '__main__':
    device = torch.device("cpu")
    torch.set_num_threads(NUM_PROCESSES)

    m = NNModel(device, int(time.time()))

    try:
        training_loader = torch.load("training_loaders/training_loader.pt", map_location=device)
        exp_inps = training_loader.dataset.inps
        exp_outs = training_loader.dataset.outs

        log_r = open("log_q.csv", 'r').readlines()
        epoch = int(log_r[len(log_r) - 1].split(', ')[2])
        print("Restarting from epoch {}.".format(epoch))
        log_q = open("log_q.csv", 'a')

    except FileNotFoundError:
        epoch = 0
        log_q = open("log_q.csv", 'w')
        log_q.write("Time, Seed, Epoch, Fitness\n") # header
        exp_inps = []
        exp_outs = []

    while True:
        epoch += 1
        m.seed = int(time.time())

        results = m.train()
        exp_inps.extend(results['exp_inps'])
        exp_outs.extend(results['exp_outs'])

        training_data = QDataset(
            inps=exp_inps,
            outs=exp_outs
        )

        log_q.write("{}, {}, {}, {}\n".format(
            time.asctime(),
            m.seed,
            epoch,
            results['q_fitness']
        ))
        log_q.flush()

        if not epoch % 5:
            training_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=16,
                shuffle=True,
                num_workers=1
            )
            torch.save(training_loader, "training_loaders/training_loader-{}.pt".format(epoch))
            os.system("cp training_loaders/training_loader-{}.pt training_loaders/training_loader.pt".format(epoch)) # training_loader.pt = most recent

        print("Epoch {}: Q-Learning Fitness = {}".format(
            epoch,
            results['q_fitness'],
        ))
