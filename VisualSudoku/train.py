import numpy as np
import torch
import torch.nn.functional as F
from models import SudokuModel
from parse_data import get_datasets, get_MNIST_dataset
from torch.utils.data import DataLoader
from utils import confusion_matrix_MNIST
import time

BATCH_SIZE = 1024
BATCH_SIZE_VAL = 10000000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 15000
EPOCHS_MAXSAT = 3000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_dataloaders = []
valid_dataloaders = []
test_dataloaders = []
for split in range(1, 12):
    # Load data
    td, vd, ted = get_datasets(split, basepath='.', use_negative_train=False)
    train_dataloaders.append(DataLoader(td, batch_size=BATCH_SIZE, shuffle=True))
    valid_dataloaders.append(DataLoader(vd, batch_size=BATCH_SIZE_VAL, shuffle=True))
    test_dataloaders.append(DataLoader(ted, batch_size=BATCH_SIZE_VAL, shuffle=True))

accuracies = []
accuracies_MNIST = []
times = []
for SPLIT in range(10):
    starting_time = time.time()
    MNIST_dataset = get_MNIST_dataset(split=SPLIT + 1)
    MNIST_dataloader = DataLoader(MNIST_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True)


    print(f'Starting with split {SPLIT}')
    # Model
    model = SudokuModel().to(DEVICE)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    criterion = torch.nn.BCEWithLogitsLoss()

    print('Start learning...')

    # Learn
    for e in range(EPOCHS):
        for i, (puzzle, _) in enumerate(train_dataloaders[SPLIT]):
            puzzle = puzzle.to(DEVICE)
            if e < EPOCHS_MAXSAT:
                target = torch.ones((puzzle.shape[0], 14823)).to(DEVICE)
            else:
                target = torch.ones((puzzle.shape[0], )).to(DEVICE)


            optimizer.zero_grad()
            truth = model(puzzle)

            if e < EPOCHS_MAXSAT:
                loss = criterion(truth, target)
            else:
                loss = criterion(torch.min(truth, dim=1)[0], target)
            loss.backward()
            optimizer.step()

        if e % 100 == 0:
            print(f'Epoch {e}, loss {loss.item()}')
            model.eval()
            # Test accuracy
            with torch.no_grad():
                for i, (puzzle, target) in enumerate(test_dataloaders[SPLIT]):
                    puzzle = puzzle.to(DEVICE)
                    target = target.to(DEVICE)

                    truth = torch.where(torch.min(model(puzzle), 1)[0] > 0., 1., 0.).to(DEVICE)
                    print(torch.sum(truth))

                    accuracy = torch.sum(torch.eq(target, truth)).float() / target.shape[0]
                    print(f'Accuracy: {accuracy}')
            model.train()

        if e % 500 == 0:
            model.eval()
            cm, accuracy_MNIST = confusion_matrix_MNIST(model.MNIST_model, MNIST_dataloader, DEVICE)
            print(cm)
            print(f'MNISt accuracy: {accuracy_MNIST}')
            model.train()

    trainig_time = time.time() - starting_time
    times.append(trainig_time)

    # Test accuracy

    model.eval()
    with torch.no_grad():
        for i, (puzzle, target) in enumerate(test_dataloaders[SPLIT]):
            puzzle = puzzle.to(DEVICE)
            target = target.to(DEVICE)

            truth = torch.where(torch.min(model(puzzle), 1)[0] > 0., 1., 0.).to(DEVICE)
            print(f'Number of predicted valid sudokus: {torch.sum(truth)}')

            accuracy = torch.sum(torch.eq(target, truth)).float() / target.shape[0]
            print(f'Accuracy: {accuracy}')

            cm, accuracy_MNIST = confusion_matrix_MNIST(model.MNIST_model, MNIST_dataloader, DEVICE)
            print(cm)
            print(f'MNISt accuracy: {accuracy_MNIST}')
    model.train()

    print(f'Accuracy for split {SPLIT} is {accuracy}. The MNIST accuracy is {accuracy_MNIST}')
    print(f'End of split {SPLIT}. Time: {trainig_time}')
    accuracies.append(accuracy)
    accuracies_MNIST.append(accuracy_MNIST)

print(f'Required training times: {times}')
print(f'Averaged time: {np.array(times).mean()}')
numpy_accuracies = np.array([a.cpu() for a in accuracies])
print(f'Accuracies: {numpy_accuracies}')
print(f'Mean accuracy: {numpy_accuracies.mean()}')
print(f'Std accuracy: {numpy_accuracies.std()}')
numpy_accuracies_MNIST = np.array([a for a in accuracies_MNIST])
print(f'Accuracies MNIST: {numpy_accuracies_MNIST}')
print(f'Mean accuracy MNIST: {numpy_accuracies_MNIST.mean()}')
print(f'Std accuracy MNIST: {numpy_accuracies_MNIST.std()}')