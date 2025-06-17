import torch
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform, AffineTransform
import math
from torch.distributions import Gumbel


class MNIST_Net(torch.nn.Module):
    def __init__(self, N=9, channels=1):
        super(MNIST_Net, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 6, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True)
        )
        self.classifier_mid = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(84, N)
        )
        self.channels = channels

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            print('init conv2, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Linear):
            print('init Linear, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier_mid(x)
        x = self.classifier(x)
        return x


def logistic_distribution(mean=0.0, std_dev=1.0):
    scale = std_dev * math.sqrt(3) / math.pi

    base_distribution = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=mean, scale=scale)]
    logistic = TransformedDistribution(base_distribution, transforms)

    return logistic


def check(ti, tj):
    '''Check if ti and tj are different

    :param ti: truth values of first cell
    :param tj: truth values of second cell
    :return: vector of truth values representing satisfaction of the constraints for each pair of digits
    '''
    return torch.max(-ti, -tj)


class SudokuModel(torch.nn.Module):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.MNIST_model = MNIST_Net()
        self.distribution = Gumbel(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        images = x.view(-1, 1, 28, 28)
        boards = self.MNIST_model(images).view(batch_size, 9, 9, 9)

        if self.training:
            sampled_boards = boards + self.distribution.sample(sample_shape=boards.shape).to(boards.device)
        else:
            sampled_boards = boards


        # Normalization
        top2_x, idx_x = torch.topk(sampled_boards, 2, dim=-1)
        scaling_factor_x = (top2_x[:, :, :, 0] + top2_x[:, :, :, 1]) / 2
        sampled_boards = sampled_boards - scaling_factor_x.unsqueeze(-1)


        # Blocks definition
        blocks = sampled_boards.view(batch_size, 3, 3, 3, 3, 9).transpose(2, 3).reshape(batch_size, 9, 9, 9)

        #  === Rules: every value should be at least in one cell in a row, column and block ===
        rows_at_least_one = torch.max(sampled_boards, dim=2)[0]
        cols_at_least_one = torch.max(sampled_boards, dim=1)[0]
        blocks_at_least_one = torch.max(blocks, dim=2)[0]

        # === Rules: every pairs of cells in a row, column and block should have different values ===

        # Meshgrid to get pairs of indices
        ix, iy = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
        identity_mask = ix != iy
        ix_pairs = ix[identity_mask]
        iy_pairs = iy[identity_mask]

        # Rows and columns constraints
        rows_different = check(sampled_boards[:, :, ix_pairs, :], sampled_boards[:, :, iy_pairs, :])
        cols_different = check(sampled_boards[:, ix_pairs, :, :], sampled_boards[:, iy_pairs, :, :])

        # Blocks constraints
        block_mask_1 = ix % 3 != iy % 3
        block_mask_2 = ix // 3 != iy // 3
        ix_pairs_block = ix[identity_mask & block_mask_1 & block_mask_2]
        iy_pairs_block = iy[identity_mask & block_mask_1 & block_mask_2]
        blocks_different = check(blocks[:, :, ix_pairs_block, :], blocks[:, :, iy_pairs_block, :])

        # All constraints
        sat = torch.cat([rows_at_least_one.view(batch_size, -1),
                         cols_at_least_one.view(batch_size, -1),
                         blocks_at_least_one.view(batch_size, -1),
                         rows_different.view(batch_size, -1),
                         cols_different.view(batch_size, -1),
                         blocks_different.view(batch_size, -1)], dim=1)

        return sat