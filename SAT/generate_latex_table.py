import os
import numpy as np

# ---------------------------------------------------------------------
# USER-SPECIFIC DATA
# ---------------------------------------------------------------------

BASE_DIR = "SAT_results"

METHODS = {
    "Product Logic": "No_noise-Product_True_0.7000",
    "Godel Logic": "No_noise-Godel_False_3.0000",
    "GT Uniform": "Godel_True_uniform_False_0.0500",
    "GT Logistic": "Godel_True_logistic_True_0.5000",
}
all_benchmarks = {
    'Uniform Random 3-SAT': ['uf20-91', 'uf50-218', 'uf75-325', 'uf100-430', 'uf125-538', 'uf150-645', 'uf200-860', 'uf225-960', 'uf250-1065'],
    'Random-3-SAT Instances and Backbone-minimal Sub-instances': ['RTI_k3_n100_m429', 'BMS_k3_n100_m429'],
    'Random 3-SAT Controlled Backbone': [
              'CBS_k3_n100_m403_b10', 'CBS_k3_n100_m403_b30', 'CBS_k3_n100_m403_b50', 'CBS_k3_n100_m403_b70',
              'CBS_k3_n100_m403_b90', 'CBS_k3_n100_m411_b10', 'CBS_k3_n100_m411_b30', 'CBS_k3_n100_m411_b50',
              'CBS_k3_n100_m411_b70', 'CBS_k3_n100_m411_b90', 'CBS_k3_n100_m418_b10', 'CBS_k3_n100_m418_b30',
              'CBS_k3_n100_m418_b50', 'CBS_k3_n100_m418_b70', 'CBS_k3_n100_m418_b90', 'CBS_k3_n100_m423_b10',
              'CBS_k3_n100_m423_b30', 'CBS_k3_n100_m423_b50', 'CBS_k3_n100_m423_b70', 'CBS_k3_n100_m423_b90',
              'CBS_k3_n100_m429_b10', 'CBS_k3_n100_m429_b30', 'CBS_k3_n100_m429_b50', 'CBS_k3_n100_m429_b70',
              'CBS_k3_n100_m429_b90', 'CBS_k3_n100_m435_b10', 'CBS_k3_n100_m435_b30', 'CBS_k3_n100_m435_b50',
              'CBS_k3_n100_m435_b70', 'CBS_k3_n100_m435_b90', 'CBS_k3_n100_m441_b10', 'CBS_k3_n100_m441_b30',
              'CBS_k3_n100_m441_b50', 'CBS_k3_n100_m441_b70', 'CBS_k3_n100_m441_b90', 'CBS_k3_n100_m449_b10',
              'CBS_k3_n100_m449_b30', 'CBS_k3_n100_m449_b50', 'CBS_k3_n100_m449_b70', 'CBS_k3_n100_m449_b90'
    ],
    'Flat Graph Colouring': ['flat30-60', 'flat50-115', 'flat75-180', 'flat100-239', 'flat200-479'],
    'Morphed Graph Colouring': ['SW100-8-2', 'SW100-8-3', 'SW100-8-4', 'SW100-8-5', 'SW100-8-6', 'SW100-8-7', 'SW100-8-8','SW100-8-p0', 'sw100-8-lp0-c5', 'sw100-8-lp1-c5'],
    'Planning': ['blocksworld', 'logistics'],
    'All interval series': ['ais'],
    'Quasigroup': ['QG'],
    'Bounded model checking': ['bmc'],
    'DIMACS': ['ai', 'ai2', 'ai3', 'f',
              'inductive-inference',
              'parity'],
    'Beijing': ['Bejing']
}

all_benchmarks_short_names = {
    'Uniform Random 3-SAT': 'UF',
    'Random-3-SAT Instances and Backbone-minimal Sub-instances': 'RTI/BMS',
    'Random 3-SAT Controlled Backbone': 'CBS',
    'Flat Graph Colouring': 'FLAT',
    'Morphed Graph Colouring': 'SW',
    'Planning': 'PLANNING',
    'All interval series': 'AIS',
    'Quasigroup': 'QG',
    'Bounded model checking': 'BMC',
    'DIMACS': 'DIMACS',
    'Beijing': 'BEIJING'
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def latex_escape(s: str) -> str:
    """
    Escape underscores for LaTeX.
    """
    return s.replace("_", r"\_")


def format_mean_std(mean_val, std_val, is_bold=False):
    """
    Returns a LaTeX string for mean ± std, with two decimal places.
    If is_bold=True, wraps the mean in \mathbf{...}.
    """
    if mean_val is None or std_val is None:
        return "---"
    if is_bold:
        return f"$\\mathbf{{{mean_val:.1f}}} \\pm \\mathbf{{{std_val:.1f}}}$"
    else:
        return f"${mean_val:.1f} \\pm {std_val:.1f}$"


def compute_metrics(benchmark, method_label):
    """
    Compute (s_mean, s_std, b_mean, b_std) for a single (benchmark, method).
    Returns None if missing files.
    """
    method_folder = METHODS[method_label]
    path = os.path.join(BASE_DIR, benchmark, method_folder)

    num_solved_path = os.path.join(path, "num_solved.npy")
    if not os.path.isfile(num_solved_path):
        return None

    num_solved = np.load(num_solved_path)
    s_mean = float(np.mean(num_solved))
    s_std = float(np.std(num_solved))

    x = (num_solved != 0).astype(float)
    b_mean = float(np.mean(x))
    b_std = float(np.std(x))

    return (s_mean, s_std, b_mean, b_std)


# ---------------------------------------------------------------------
# PRECOMPUTE RESULTS FOR ALL (BENCHMARK, METHOD)
# ---------------------------------------------------------------------

results = {}
for domain, bench_list in all_benchmarks.items():
    for bench in bench_list:
        for method_name in METHODS.keys():
            metrics = compute_metrics(bench, method_name)
            results[(bench, method_name)] = metrics


# ---------------------------------------------------------------------
# FIRST TABLE: GROUP BY DOMAIN, PERCENTAGES, NO I COLUMN
# ---------------------------------------------------------------------
def start_table():
    """
    Prints the LaTeX preamble for the table.
    """
    print(r"\begin{table*}[h!]")
    print(r"    \centering")
    print(r"    \resizebox{\textwidth}{!}{%")
    # 1 col for Benchmark + 4 methods * 2 cols (S,B) = 1 + (4*2) = 9
    print(r"\begin{tabular}{|l|cc|cc|cc|cc|}")
    print(r"\hline")
    print(r"\textbf{Benchmark}"
          r" & \multicolumn{2}{c|}{\textbf{Product Logic}}"
          r" & \multicolumn{2}{c|}{\textbf{Godel Logic}}"
          r" & \multicolumn{2}{c|}{\textbf{GT Uniform}}"
          r" & \multicolumn{2}{c|}{\textbf{GT Logistic}} \\")
    print(r"\hline")
    # Include the % symbol in headers
    print(r" & \textbf{S(\%)} & \textbf{B(\%)}"
          r" & \textbf{S(\%)} & \textbf{B(\%)}"
          r" & \textbf{S(\%)} & \textbf{B(\%)}"
          r" & \textbf{S(\%)} & \textbf{B(\%)} \\")
    print(r"\hline")


def end_table(final_caption=False):
    """
    Closes the LaTeX table.
    If final_caption=True, prints the final caption, otherwise a shorter one.
    """
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"    }")
    if final_caption:
        print(
            r"    \caption{Comparative table showing results on SATLIB benchmarks for four methods: "
            r"Product Logic, Godel Logic, \godel Trick Uniform, and \godel Trick Logistic. "
            r"Columns: S stands for Sample Solved (mean $\pm$ std of number of solved samples), "
            r"B stands for Best Solution f(mean $\pm$ std of number of problem solved by keeping the best"
            r" results among the samples). Highest S and B in bold."
            r"``---'' on Bejing is due to the reaching of the time limit of 12 hours. Continue in the next page.}"
        )
    else:
        print(r"\caption{Comparative table showing results on SATLIB benchmarks for four methods. "
              r"Continue in the next page.}")
    print(r"    \label{tab:methods_comparison_full}")
    print(r"\end{table*}")


# We'll print the table in chunks of 50 data rows (domain rows + benchmark rows).
start_table()

line_count = 0
row_color_idx = 0

for domain in all_benchmarks:
    if domain == 'Flat Graph Colouring':
        end_table()
        start_table()
        line_count = 0
    domain_escaped = latex_escape(domain)

    # Domain row (spanning 9 columns)
    # \multicolumn{9}{|c|}{domain_escaped}
    # We'll treat it like a "data row" so it counts toward line_count.

    print(r"\hline")
    print(r"\multicolumn{9}{|c|}{" + domain_escaped + r"} \\")
    print(r"\hline")
    line_count += 1

    # Now each benchmark in this domain
    for bench in all_benchmarks[domain]:

        # Optional row color
        if row_color_idx % 2 == 1:
            print(r"\rowcolor{gray!20}")
        row_color_idx += 1
        line_count += 1

        bench_escaped = latex_escape(bench)

        # Collect metrics for each method
        method_data = []
        for method_name in ["Product Logic", "Godel Logic", "GT Uniform", "GT Logistic"]:
            metrics = results.get((bench, method_name), None)
            # metrics => (s_mean, s_std, b_mean, b_std)
            if metrics is None:
                method_data.append({
                    "s_mean": None, "s_std": None,
                    "b_mean": None, "b_std": None,
                    "valid": False
                })
            else:
                (s_mean, s_std, b_mean, b_std) = metrics
                method_data.append({
                    "s_mean": s_mean, "s_std": s_std,
                    "b_mean": b_mean, "b_std": b_std,
                    "valid": True
                })

        # Find max S, B among valid entries
        s_values = [md["s_mean"] for md in method_data if md["s_mean"] is not None]
        b_values = [md["b_mean"] for md in method_data if md["b_mean"] is not None]
        max_s = max(s_values) if s_values else 0.0
        max_b = max(b_values) if b_values else 0.0

        # If max_s == 0 and max_b == 0 => skip bold
        all_zero = (max_s == 0.0 and max_b == 0.0)

        row_cells = [bench_escaped]

        for md in method_data:
            if not md["valid"]:
                row_cells.extend(["---", "---"])
                continue
            s_mean, s_std = md["s_mean"], md["s_std"]
            b_mean, b_std = md["b_mean"], md["b_std"]
            # Convert to percentages
            s_mean_100 = s_mean * 100
            s_std_100 = s_std * 100
            b_mean_100 = b_mean * 100
            b_std_100 = b_std * 100

            s_bold = (not all_zero) and (s_mean == max_s)
            b_bold = (not all_zero) and (b_mean == max_b)

            s_str = format_mean_std(s_mean_100, s_std_100, is_bold=s_bold)
            b_str = format_mean_std(b_mean_100, b_std_100, is_bold=b_bold)
            row_cells.extend([s_str, b_str])

        print(" & ".join(row_cells) + r" \\")

# End the first table (final_caption=True)
end_table(final_caption=True)

# ---------------------------------------------------------------------
# SECOND TABLE: ONLY GT METHODS, AGGREGATED BY DOMAIN (mean ± std)
# ---------------------------------------------------------------------
# We'll compute the average (S,B) for each domain, for each GT method,
# across all benchmarks in that domain.

print("\n\n")  # Some spacing between tables


def start_table_2():
    """
    Prints the LaTeX preamble for the second table (only 2 methods => 1 col domain + 2*(2 cols) = 5).
    """
    print(r"\begin{table}[h!]")
    print(r"    \centering")
    print(r"    \resizebox{235px}{!}{%")
    print(r"\begin{tabular}{|l|cc|cc|}")
    print(r"\hline")
    print(r"\textbf{Domain}"
          r" & \multicolumn{2}{c|}{\textbf{GT Uniform}}"
          r" & \multicolumn{2}{c|}{\textbf{GT Logistic}} \\")
    print(r"\hline")
    print(r" & \textbf{S(\%)} & \textbf{B(\%)}"
          r" & \textbf{S(\%)} & \textbf{B(\%)} \\")
    print(r"\hline")


def end_table_2():
    """
    Closes the second table.
    """
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"    }")
    print(
        r"    \caption{Comparative table on SATLIB benchmarks by domain, for two methods: "
        r"\godel Trick Uniform, and \godel Trick Logistic. "
        r"Columns: S stands for Sample Solved (mean $\pm$ std of number of solved samples), "
        r"B stands for Best Solution f(mean $\pm$ std of number of problem solved by keeping the best"
        r" results among the samples). Highest S and B in bold.}"
    )
    print(r"    \label{tab:methods_comparison}")
    print(r"\end{table}")


start_table_2()
row_color_idx = 0

GT_METHODS = ["GT Uniform", "GT Logistic"]

for domain in all_benchmarks:
    # Alternate row color
    if row_color_idx % 2 == 1:
        print(r"\rowcolor{gray!20}")
    row_color_idx += 1

    # For LaTeX, escape underscores, etc.
    domain_escaped = latex_escape(domain)

    # We compute the aggregated (s_m, b_m) and (s_s, b_s) for each method
    # by taking the mean ± std over all benchmarks in 'domain'.
    method_data = []
    for method_name in GT_METHODS:
        sb_pairs = []
        for bench in all_benchmarks[domain]:
            m = results.get((bench, method_name), None)  # (s_mean, s_std, b_mean, b_std)
            if m is not None:
                s_mean, s_std, b_mean, b_std = m
                sb_pairs.append((s_mean, b_mean))
        if len(sb_pairs) == 0:
            # No data => store None
            method_data.append({
                "method": method_name,
                "s_m": None, "s_s": None,
                "b_m": None, "b_s": None,
                "valid": False
            })
        else:
            arr = np.array(sb_pairs)  # shape [N, 2] => columns: s_mean, b_mean
            s_m = arr[:, 0].mean()
            b_m = arr[:, 1].mean()
            s_s = arr[:, 0].std()
            b_s = arr[:, 1].std()
            # Multiply by 100 for percentages
            s_m *= 100
            s_s *= 100
            b_m *= 100
            b_s *= 100
            method_data.append({
                "method": method_name,
                "s_m": s_m, "s_s": s_s,
                "b_m": b_m, "b_s": b_s,
                "valid": True
            })

    # Determine max S, B across the two methods (only for valid ones)
    s_vals = [md["s_m"] for md in method_data if md["s_m"] is not None]
    b_vals = [md["b_m"] for md in method_data if md["b_m"] is not None]

    max_s = max(s_vals) if s_vals else 0.0
    max_b = max(b_vals) if b_vals else 0.0

    # If *both* are zero => skip bold entirely
    all_zero = (max_s == 0.0 and max_b == 0.0)

    # Build the row (domain + 2*(S,B) cells)
    row_cells = [all_benchmarks_short_names[domain]]
    for md in method_data:
        if not md["valid"]:
            row_cells.extend(["---", "---"])
        else:
            s_m = md["s_m"]
            s_s = md["s_s"]
            b_m = md["b_m"]
            b_s = md["b_s"]
            # Bold if not all_zero and equals max
            s_bold = (not all_zero) and (s_m == max_s)
            b_bold = (not all_zero) and (b_m == max_b)

            s_str = format_mean_std(s_m, s_s, is_bold=s_bold)
            b_str = format_mean_std(b_m, b_s, is_bold=b_bold)
            row_cells.extend([s_str, b_str])

    print(" & ".join(row_cells) + r" \\")

end_table_2()
