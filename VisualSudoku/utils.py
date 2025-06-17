import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def normalizeMNIST(images):
    return images.round(decimals=4)


def get_dataset():
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    mnist_test_loader = dataloader(mnist_test_data, 32)

    return mnist_test_loader


def confusion_matrix_MNIST(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            data = normalizeMNIST(data)
            outputs = model(data.unsqueeze(1)).to(device)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    maximum_cols_indeces = []
    for i in range(9):
        maximum_cols_indeces.append(cm[i, :].argmax())

    # Permute matrix
    cm = cm[:, maximum_cols_indeces]

    # Compute accuracy
    accuracy = cm.diagonal().sum() / cm.sum()

    return cm, accuracy
