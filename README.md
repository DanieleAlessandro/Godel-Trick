# Gödel Trick

**Source code for paper: “Noise to the Rescue: Escaping Local Minima in Neurosymbolic Local Search”**

If you use this software for academic research, please, cite our work using the following BibTeX:
```
@article{daniele2025noise,
  title={Noise to the Rescue: Escaping Local Minima in Neurosymbolic Local Search},
  author={Daniele, Alessandro and van Krieken, Emile},
  journal={arXiv preprint arXiv:2503.01817},
  year={2025}
}
```

---

## Installation

To install the required packages:
```bash
cd GodelTrick
python -m pip install -e .
```

For experiments with the SAT solver, you also need to install `torch_scatter`. For further details visit the [torch_scatter documentation](https://github.com/rusty1s/pytorch_scatter).

## VisualSudoku Experiments

### Dataset Download

1. Visit [https://github.com/linqs/visual-sudoku-puzzle-classification](https://github.com/linqs/visual-sudoku-puzzle-classification).
2. In the *Pre-Generated Data* section, download **Basic 9×9 MNIST**.
3. Extract the archive and copy the folder `ViSudo-PC` into `GodelTrick/VisualSudoku/`&#x20;

### Training and Evaluation

```bash
python train.py
```

## SAT Experiments

### Benchmark Download

Download any benchmark suite in DIMACS format from the SATLIB collection: [https://www.cs.ubc.ca/\~hoos/SATLIB/benchm.html](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html). For instance `uf20-91.tar.gz`.

### Folder Layout

```bash
mkdir -p SAT/SAT_instances
# Example for uf20-91
mkdir -p SAT/SAT_instances/uf20-91
# extract .cnf files here
```

You can place multiple benchmarks side by side (`uf50-218`,  …). Each directory name becomes a *benchmark* identifier.

**NB:** GT has been tested only on "all sat" instances.

### Running

The main script is `SAT/run.py`. Below are two quick recipes:

**1️⃣ Run with default parameters (only mandatory flag is the benchmark folder):**

```bash
cd SAT
python run.py --benchmarks uf20-91
```

**2️⃣ Custom run on a single benchmark with Gödel semantics, uniform noise, and explicit hyper‑parameters:**

````bash
cd SAT
python run.py \
  --benchmarks uf20-91 \
  --semantics Godel \
  --noise \
  --distribution uniform \
  --batch_size 1000 \
  --learning_rate 0.05 \
  --device cuda
````

Key arguments:

| Argument                 | Default    | Description                                                           |
| ------------------------ | ---------- | --------------------------------------------------------------------- |
| `--benchmarks`           | *required* | Folder names in `SAT/SAT_instances/`                                  |
| `--semantics`            | `Godel`    | The semantics used (`Godel`, `Product`, `Lukasiewicz`, `Log‑Product`) |
| `--noise` / `--no-noise` | `--noise`  | Add input noise during optimisation                                   |
| `--distribution`         | `uniform`  | Noise distribution (`uniform`, `logistic`, `none`)                    |
| `--sigmoid`              | `False`    | Enable BCELoss objective                                              |
| `--auto_pad`             | `False`    | Zero‑pad clauses to the longest formula in the batch                  |
| `--number_of_steps`      | `500000`   | Optimisation iterations                                               |

Run `python run.py -h` for the full list.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.