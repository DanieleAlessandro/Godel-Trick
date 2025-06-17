from torch.utils.data import Dataset, DataLoader
import os
import torch
import utils as u
import math

class SATDataset(Dataset):
    def __init__(self, benchmark_dir: str, auto_pad: bool = True):
        """
        Args:
            benchmark_dir: Directory containing SAT problem files
        """
        self.file_paths = []
        self.problems = []

        # Load all problems from the benchmark directory
        for file_name in os.listdir(benchmark_dir):
            if file_name.startswith("."):
                continue
            file_path = os.path.join(benchmark_dir, file_name)
            self.file_paths.append(file_path)

            # Parse the CNF file and store the problem data
            num_vars, gather_indices, signs, scatter_indices = u.parse_cnf_file(
                file_path
            )
            self.problems.append((num_vars, gather_indices, signs, scatter_indices))

        if auto_pad:
            max_literals = max(problem[1].shape[0] for problem in self.problems)
            target_clauses = max(problem[3].max() for problem in self.problems)
            if not all(
                problem[1].shape[0] == max_literals for problem in self.problems
            ):
                print("Padding problems to have the same number of literals")
                # Padding is performed by adding an additional variable that is always 1
                for i in range(len(self.problems)):
                    problem = self.problems[i]
                    num_vars, gather_indices, signs, scatter_indices = problem
                    diff_literals = max_literals - gather_indices.shape[0]
                    start_new_clause = min(max(scatter_indices) + 1, target_clauses)
                    self.problems[i] = (
                        # Add one variable for the padding
                        num_vars + 1,
                        # Gathering: Only gather the new variable
                        torch.cat(
                            [gather_indices, torch.full((diff_literals,), num_vars)]
                        ),
                        # Sign: All positive
                        torch.cat([signs, torch.full((diff_literals,), 1)]),
                        # Scatter: All clauses between max_clause + 1 and max_clause + 1 + target_clauses, and pad further at the end if needed, AND over positive new variable literal
                        torch.cat(
                            [scatter_indices, torch.arange(start_new_clause, target_clauses), torch.full((diff_literals - target_clauses + start_new_clause,), target_clauses)]
                        ),
                    )

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


def get_sat_dataloader(benchmark_dir: str, batch_size: int = 1, shuffle: bool = False, auto_pad: bool = True):
    """Helper function to create a SAT problem dataloader

    Args:
        benchmark_dir: Directory containing SAT problem files
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
    Returns:
        DataLoader for SAT problems
    """
    dataset = SATDataset(benchmark_dir, auto_pad=auto_pad)
    return DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle
    )
