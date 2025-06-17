import torch


def parse_cnf_file(file_path: str):
    """Parse a CNF file and return the positive literals and their signs.

    :param file_path: path of the .cnf file
    :return: a tensor of positive literals (for gather), a tensor with their signs,
    and a tensor with the clause indeces (for scatter)
    """
    gather_indices = []
    signs = []
    scatter_indices = []
    current_clause = 0
    number_of_variables = 0

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("c") or line.startswith("%"):
                continue

            if line.startswith("p"):
                number_of_variables = int(line.split()[2])
                continue

            clause = [int(literal) for literal in line.split() if literal != "0"]
            gather_indices += [abs(literal) - 1 for literal in clause]
            signs += [1. if literal > 0 else -1. for literal in clause]
            scatter_indices += [current_clause] * len(clause)
            current_clause += 1

    return (number_of_variables,
            torch.Tensor(gather_indices).long(),
            torch.Tensor(signs),
            torch.Tensor(scatter_indices).long())


# If main
if __name__ == "__main__":
    nv, pl, s, ci = parse_cnf_file("SAT_instances/uf20-91/uf20-01.cnf")
    print(nv)
    print(pl)
    print(s)
    print(ci)
