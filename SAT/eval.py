import os
import numpy as np


for benchmark in os.listdir("SAT_results"):
    print('====================================================')
    print(f'Results for {benchmark}')
    results_path = os.path.join("SAT_results", benchmark)

    files_names = os.listdir(results_path)

    for file_name in files_names:
        if file_name.endswith(".npy") and not file_name.startswith("time") and not file_name.startswith("iterations"):
            x = np.load(os.path.join(results_path, file_name))
            mean = np.mean(x)
            std = np.std(x)

            print('====================================================')
            print(f'Results for {file_name}')
            print(f'Mean: {mean}')
            print(f'Standard deviation: {std}')
            number_of_solutions = np.sum(x > 0.0)
            print(f'Number of solutions found: {number_of_solutions}')

            x2 = np.load(os.path.join(results_path, 'time_spent_' + file_name))
            print(f'Mean time spent: {np.mean(x2)}')

            if os.path.join(results_path, 'iterations_' + file_name) in files_names:
                x3 = np.load(os.path.join(results_path, 'iterations_' + file_name))
                print(f'Mean iterations to reach an optimum: {np.mean(x3)}')

                # Check if number of solutions found is correct
                number_of_solutions_2 = np.sum(x3 != 50000.0)
                print(f'Number of solutions found (2): {number_of_solutions_2}')