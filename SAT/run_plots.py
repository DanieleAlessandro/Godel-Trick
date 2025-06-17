import torch
import os
from random import choices
from SAT.madgrad import MADGRAD
import numpy as np
import argparse

from models import SATFormula
from distributions import uniform_distribution, logistic_distribution, clip_params

import time

parser = argparse.ArgumentParser(description='SAT problem solver')
parser.add_argument('--benchmark', type=str, default='uf20-91', help='Benchmark name')
parser.add_argument('--config', type=int, default=0, help='Configuration number')

# File paths settings
benchmark = parser.parse_args().benchmark
input_path = os.path.join("SAT_instances", benchmark)
if not os.path.exists("SAT_results_plot"):
    os.makedirs("SAT_results_plot")
if not os.path.exists(os.path.join("SAT_results_plot", benchmark)):
    os.makedirs(os.path.join("SAT_results_plot", benchmark))
output_path = os.path.join("SAT_results_plot", benchmark)

number_of_files = len(os.listdir(input_path))
files_names = os.listdir(input_path)

# Optimization settings
number_of_steps = 20000
learning_rate = 0.05
momentum = 0.5
number_of_samples = 100

config = parser.parse_args().config

# The noise is sampled only once, but the matrix is k times bigger and the position inside the matrix is chosen
# randomly at each step
k = 5

# Models configurations
# Product, no noise, use sigmoid
# Lukasiewicz, no noise, use sigmoid
# Godel, no noise, use sigmoid (standard Godel)
# Godel, no noise, no sigmoid (quasi standard Godel)
# Godel, noise, uniform distribution, use sigmoid
# Godel, noise, logistic distribution, use sigmoid
# Godel, noise, uniform distribution, no sigmoid
# Godel, noise, logistic distribution, no sigmoid
semantics_list = ['Product', 'Lukasiewicz', 'Godel', 'Godel', 'Godel', 'Godel', 'Godel', 'Godel']
noise_list = [False, False, False, False, True, True, True, True]
distribution_list = [None, None, None, None,
                     uniform_distribution, logistic_distribution, uniform_distribution, logistic_distribution]
use_sigmoid_list = [True, True, True, False, True, True, False, False]

with open(os.path.join(output_path, "hyperparameters.txt"), 'w') as f:
    f.write(f"Number of steps: {number_of_steps}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Momentum: {momentum}\n")
    f.write(f"Number of files: {number_of_files}\n")
    f.write(f"Files names: {files_names}\n")


semantics = semantics_list[config]
use_noise = noise_list[config]
distribution = distribution_list[config]
use_sigmoid = use_sigmoid_list[config]

# for semantics, use_noise, distribution, use_sigmoid in zip(semantics_list, noise_list,

if use_noise:
    print('semantics:', semantics, 'noise:', use_noise, 'distribution:',
          distribution.__name__.split('_')[0], 'use_sigmoid:', use_sigmoid)
    output_file_name = f"{semantics}_{use_noise}_{distribution.__name__}_{use_sigmoid}"
else:
    print('Standard fuzzy logic. Semantics: ', semantics, ' Use sigmoid: ', use_sigmoid)
    output_file_name = f"{semantics}_{use_sigmoid}"

output_file = os.path.join(output_path, output_file_name)

# Initialize the lists to store the results
results = []
solved = 0
total = 0
count = None
time_spent = []
starting_indices = None
current_noise = None
threshold = 0.0
target = None
if use_sigmoid:
    threshold = 0.5
    target = torch.ones([number_of_samples]).to('cuda')  # loss becomes the BCELoss

# Cycle over all the instances of SAT problems
for instance_number, file_path in enumerate(os.listdir(input_path)):
    results_instance = []

    f = SATFormula(os.path.join(input_path, file_path), number_of_samples, semantics, use_sigmoid).to('cuda')

    print(f'Instance {instance_number + 1}/{number_of_files}')
    print(f'File: {file_path}')
    print('===================================')

    optimizer = MADGRAD([f.propositions], lr=learning_rate, momentum=momentum)

    if use_noise:
        noise = distribution().sample(
            [number_of_samples * k, f.propositions.shape[1]]).to('cuda')
        starting_indices = choices(range(0, number_of_samples * (k - 1), 1),
                                   k=number_of_steps)
    else:
        noise = None
        current_noise = 0.0

    start = time.time()

    # Optimization loop
    for i in range(number_of_steps):
        optimizer.zero_grad()

        if noise is not None:
            current_noise = noise[starting_indices[i]:starting_indices[i] + number_of_samples, :]

        grounded_clauses = f(noise=current_noise)

        if use_sigmoid:
            loss = torch.nn.BCELoss()(grounded_clauses, target)
        else:
            loss = -torch.sum(grounded_clauses)

        loss.backward()
        optimizer.step()

        if distribution == uniform_distribution:
            clip_params(f.propositions, -1., 1.)

        if i % 100 == 0:
            solved = f(discretize=True)
            results_instance.append(solved.detach().cpu().numpy())

            if torch.sum(solved) == number_of_samples:
                results_instance += [results_instance[-1]] * ((number_of_steps // 100) - (i // 100) - 1)
                break

    end = time.time()
    time_spent.append(end - start)
    results.append(np.column_stack(results_instance))
    print(f'Time spent: {end - start}')
    print(f'Solved: {torch.sum(solved)}/{number_of_samples}')

time_spent_array = np.array(time_spent)
print(f'Total time spent: {np.sum(time_spent_array)}')
print(f'Mean time spent: {np.mean(time_spent_array)}')
final_results = np.stack(results, axis=-1)
np.save(output_file, final_results)
np.save(os.path.join(output_path, f"time_spent_{output_file_name}"), time_spent_array)

