import argparse
import os
import itertools


def run_experiments(nodes_list, densities_list, problem):
    """
    Runs script_rl.py for all Cartesian product combinations of nodes and
    densities.

    Args:
        nodes_list (list): A list of integer values for 'n' (number of nodes).
        densities_list (list): A list of float values for 'p' (graph density).
    """
    print("Starting experiment run...")
    print(f"Nodes to test (n): {nodes_list}")
    print(f"Densities to test (p): {densities_list}\n")

    combinations = list(itertools.product(nodes_list, densities_list))

    print(f"Total combinations to run: {len(combinations)}\n")

    for n_val, p_val in combinations:
        command = (
            f"python3.9 script_rl.py "
            f"--problem {problem} "
            f"-a DQN "
            f"-n {n_val} "  
            f"--delta_n {n_val} "
            f"-p {p_val} "
            f"--no_attr "
        )

        print(f"Executing command: {command}")

        exit_code = os.system(command)

        if exit_code == 0:
            print(f"Command for n={n_val}, p={p_val} completed successfully.\n")
        else:
            print(f"Command for n={n_val}, p={p_val} FAILED with exit code "
                  f"{exit_code}.\n")
            break

    print("All experiments finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RL experiments for various combinations of graph "
                    "nodes (n) and densities (p). "
    )

    problems = {'mvc', 'mds'}
    parser.add_argument(
        '--problem',
        default='mvc',
        choices=problems,
        help='the CO to train.'
    )

    # Argument for 'n' (number of nodes)
    # nargs='+' means one or more arguments are expected, collected into a list
    parser.add_argument(
        '--nodes',
        type=int,
        nargs='+',
        required=True,
        help='A space-separated list of integer values for the number of '
             'nodes (n). '
    )

    # Argument for 'p' (graph density)
    parser.add_argument(
        '--densities',
        type=float,
        nargs='+',
        required=True,
        help='A space-separated list of float values for the graph density (p).'
    )

    args = parser.parse_args()

    # Call the function to run the experiments with the parsed arguments
    run_experiments(args.nodes, args.densities, args.problem)
