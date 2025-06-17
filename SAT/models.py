import torch
from torch_scatter import scatter_max, scatter_add, scatter_mul


class SATFormula(torch.nn.Module):
    # External Noise (EN)
    def __init__(self, batch_size, number_of_samples, number_of_variables, semantics, use_sigmoid, scatter_indices, signs, gather_indices):
        super().__init__()
        self.propositions = torch.nn.Parameter(torch.normal(0, 1, size=(number_of_samples, batch_size, number_of_variables)), requires_grad=True)
        self.use_sigmoid = use_sigmoid
        self.scatter_indices = scatter_indices.unsqueeze(0).expand(number_of_samples, batch_size, -1)
        self.signs = signs
        self.gather_indices = gather_indices.unsqueeze(0).expand(number_of_samples, batch_size, -1)

        if semantics == 'Godel':
            self.disjunction = lambda x: scatter_max(x, self.scatter_indices, dim=-1)[0]
            self.conjunction = lambda x: torch.min(x, dim=-1)[0]
        elif semantics == 'Product':
            self.disjunction = lambda x: 1. - scatter_mul(1. - x, self.scatter_indices, dim=-1)
            self.conjunction = lambda x: torch.prod(x, dim=-1)
        elif semantics == 'Lukasiewicz':
            self.disjunction = lambda x: torch.clamp(scatter_add(x, self.scatter_indices, dim=-1), 0, 1)
            self.conjunction = lambda x: torch.clamp(torch.sum(x, dim=-1) - (x.shape[-1] - 1.), 0, 1)
        elif semantics == 'Log-Product':
            self.disjunction = lambda x: 1. - scatter_mul(1. - x, self.scatter_indices, dim=-1)
            self.conjunction = lambda x: torch.sum(torch.log(x+1e-10), dim=-1)
        else:
            raise ValueError(f"Semantics {semantics} not supported.")

    def forward(self,
                noise=0.0,
                discretize=False) -> torch.Tensor:
        """This function computes the truth values of the grounded clauses of the SAT formula.

        Args:
            noise: the noise to add to the propositions.
            discretize: a boolean indicating whether to discretize the truth values.

        Returns:
            The truth values of the formula.
        """
        noisy_propositions = self.propositions + noise

        grounded_literals = torch.gather(noisy_propositions, -1, self.gather_indices) * self.signs
        if self.use_sigmoid:
            grounded_literals = torch.nn.functional.sigmoid(grounded_literals)

        if discretize:
            if self.use_sigmoid:
                grounded_literals = (grounded_literals > 0.5).float()
            else:
                grounded_literals = (grounded_literals > 0.0).float()

        grounded_clauses = self.disjunction(grounded_literals)

        return self.conjunction(grounded_clauses)

    def get_solution(self):
        """This function returns the solution of the SAT formula.

        Returns:
            The solution of the formula.
        """
        return self.propositions > 0.0


if __name__ == "__main__":
    # Tests

    # Godel semantics
    print('Godel semantics, preactivations')
    f = SATFormula("../test.cnf", 1, 'Godel', False).to('cuda')
    print('Propositions values:')
    print(f.propositions)
    print('Discrete propositions values:')
    print(f.get_solution())
    print('Grounded clauses values:')
    print(f(discretize=False))
    print('Discrete grounded clauses values:')
    print(f(discretize=True))

    # Godel semantics with sigmoid
    print('Godel semantics with sigmoid')
    f = SATFormula("../test.cnf", 1, 'Godel', True).to('cuda')
    print('Propositions values:')
    print(torch.sigmoid(f.propositions))
    print('Discrete propositions values:')
    print(f.get_solution())
    print('Grounded clauses values:')
    print(f(discretize=False))
    print('Discrete grounded clauses values:')
    print(f(discretize=True))

    # Product semantics
    print('Product semantics')
    f = SATFormula("../test.cnf", 1, 'Product', True).to('cuda')
    print('Propositions values:')
    print(torch.sigmoid(f.propositions))
    print('Discrete propositions values:')
    print(f.get_solution())
    print('Grounded clauses values:')
    print(f(discretize=False))
    print('Discrete grounded clauses values:')
    print(f(discretize=True))

    # Lukasiewicz semantics
    print('Lukasiewicz semantics')
    f = SATFormula("../test.cnf", 1, 'Lukasiewicz', True).to('cuda')
    print('Propositions values:')
    print(torch.sigmoid(f.propositions))
    print('Discrete propositions values:')
    print(f.get_solution())
    print('Grounded clauses values:')
    print(f(discretize=False))
    print('Discrete grounded clauses values:')
    print(f(discretize=True))
