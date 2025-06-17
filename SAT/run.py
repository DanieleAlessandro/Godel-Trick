from SAT.data import get_sat_dataloader
import torch
import os
from random import choices
from SAT.madgrad import MADGRAD
import numpy as np

from models import SATFormula
from distributions import uniform_distribution, logistic_distribution, clip_params

import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--benchmarks", type=str, nargs="+", required=True)
parser.add_argument(
    "--semantics", type=str, default="Godel", choices=["Godel", "Product", "Lukasiewicz", "Log-Product"]
)
parser.add_argument("--noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--distribution",
    type=str,
    default="uniform",
    choices=["uniform", "logistic", "none"],
)
parser.add_argument("--sigmoid", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--number_of_steps", type=int, default=500000)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--logistic_std", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--auto_pad", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
print(args)

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"All benchmarks to run: {args.benchmarks}")
# for benchmark in ['uf20-91']:
for benchmark in args.benchmarks:
    print("===================================")
    print(f"Running benchmark: {benchmark}")
    print("===================================")
    # benchmark = parser.parse_args().benchmark
    input_path = os.path.join("SAT_instances", benchmark)
    output_path = os.path.join("SAT_results", benchmark)

    if args.noise:
        print(
            f"semantics: {args.semantics}, noise: {args.noise}, distribution: {args.distribution}, sigmoid: {args.sigmoid}",
        )
        method_dir = f"{args.semantics}_{args.noise}_{args.distribution}_{args.sigmoid}_{args.learning_rate:.4f}"
    else:
        print(
            f"No noise semantics: {args.semantics}, sigmoid: {args.sigmoid}",
        )
        method_dir = f"No_noise-{args.semantics}_{args.sigmoid}_{args.learning_rate:.4f}"

    output_path = os.path.join(output_path, method_dir)
    os.makedirs(output_path, exist_ok=True)

    number_of_files = len(os.listdir(input_path))
    files_names = os.listdir(input_path)

    # Optimization settings
    number_of_steps = args.number_of_steps
    learning_rate = args.learning_rate
    momentum = 0.5
    number_of_samples = 100

    # The noise is sampled only once, but the matrix is k times bigger, and the position inside the matrix is chosen
    # randomly at each step
    k = 5

    # Models configurations
    # Product, no noise, use sigmoid
    # NO: Lukasiewicz, no noise, use sigmoid
    # NO: Godel, no noise, use sigmoid (standard Godel)
    # NO: Godel, no noise, no sigmoid (quasi standard Godel)
    # NO: Godel, noise, uniform distribution, use sigmoid
    # Godel, noise, logistic distribution, use sigmoid
    # Godel, noise, uniform distribution, no sigmoid
    # NO: Godel, noise, logistic distribution, no sigmoid

    with open(os.path.join(output_path, "hyperparameters.txt"), "w") as f:
        f.write(f"Number of steps: {number_of_steps}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Momentum: {momentum}\n")
        f.write(f"Number of files: {number_of_files}\n")
        f.write(f"Files names: {files_names}\n")

    distribution = (
        uniform_distribution
        if args.distribution == "uniform"
        else logistic_distribution
    )

    # Initialize the lists to store the results
    time_spent = []
    starting_indices = None
    current_noise = None
    target = None
    results_num_solved = []
    results_iterations = []

    dataloader = get_sat_dataloader(input_path, batch_size=args.batch_size, auto_pad=args.auto_pad)

    # Cycle over all the instances of SAT problems
    for bench_index, (num_vars, gather_indices, signs, scatter_indices) in enumerate(dataloader):
        assert torch.all(num_vars[0] == num_vars)
        num_vars = num_vars[0].item()
        gather_indices = gather_indices.to(device)
        signs = signs.to(device)
        scatter_indices = scatter_indices.to(device)

        b = gather_indices.shape[0]
        if args.sigmoid:
            target = torch.ones([number_of_samples, b]).to(
                device
            )  # loss becomes the BCELoss

        first_iteration_with_solution = torch.full(
            (b,), number_of_steps, device=device, dtype=torch.int32
        )
        number_of_solved = torch.zeros_like(first_iteration_with_solution)
        results_instance = [[] for _ in range(b)]

        f = SATFormula(
            b, number_of_samples, num_vars, args.semantics, args.sigmoid, scatter_indices, signs, gather_indices
        ).to(device)

        print(
            f"Instance {bench_index*b}-{min((bench_index + 1)*b, number_of_files)}/{number_of_files}"
        )
        print("===================================")

        optimizer = MADGRAD([f.propositions], lr=learning_rate, momentum=momentum)

        start = time.time()

        # Optimization loop
        for t in range(number_of_steps):
            optimizer.zero_grad()

            noise = 0.
            if args.noise:
                if distribution == uniform_distribution:
                    noise = distribution(device).sample([number_of_samples, b, f.propositions.shape[-1]])
                else:
                    noise = distribution(device, std_dev=args.logistic_std).sample(
                        [number_of_samples, b, f.propositions.shape[-1]])

            grounded_clauses = f(noise=noise)

            if args.sigmoid and args.semantics != 'Log-Product':
                # Note that Log-Product already implicitly implements BCELoss
                loss = torch.nn.BCELoss()(grounded_clauses, target)
            else:
                loss = -torch.sum(grounded_clauses)

            loss.backward()
            optimizer.step()

            if distribution == uniform_distribution:
                clip_params(f.propositions, -1.0, 1.0)

            if t % 100 == 0:
                solved = f(discretize=True)
                number_of_solved = torch.sum(solved, dim=0)

                first_iteration_with_solution = torch.where(
                    number_of_solved > 0,
                    torch.min(torch.tensor(t), first_iteration_with_solution),
                    first_iteration_with_solution,
                )
                if t % 10000 == 0:
                    print(
                        f"Epoch {t}: {loss.item()}, {int(number_of_solved.sum())}/{number_of_samples*b}"
                    )
                if number_of_solved.sum() == number_of_samples*b:
                    print(f"Solution found for all instances at epoch {t}")
                    break
                solved_cpu = solved.detach().cpu().numpy()
                for i in range(b):
                    results_instance[i].append(solved_cpu[:, i])

        end = time.time()
        time_spent.append(end - start)

        solved = f(discretize=True)
        number_of_solved = torch.sum(solved, dim=0).cpu().numpy()
        print(
            f"Epoch {t}: {int(number_of_solved.sum())}/{number_of_samples*b}"
        )
        first_iteration_with_solution = first_iteration_with_solution.cpu()
        for i in range(b):
            results_num_solved.append(number_of_solved[i] / number_of_samples)
            results_iterations.append(first_iteration_with_solution[i])
            results_granular = np.column_stack(results_instance[i])
            np.save(os.path.join(output_path, f"granular_{bench_index*b + i}"), results_granular)

    time_spent_array = np.array(time_spent)
    print(f"Total time spent: {np.sum(time_spent_array)}")
    print(f"Mean time spent: {np.mean(time_spent_array)}")
    np.save(
        os.path.join(output_path, f"time_spent"), time_spent_array
    )

    final_results = np.stack(results_num_solved, axis=-1)
    np.save(os.path.join(output_path, f"num_solved"), final_results)

    final_results_iterations = np.stack(results_iterations, axis=-1)
    np.save(
        os.path.join(output_path, f"iterations"),
        final_results_iterations,
    )

