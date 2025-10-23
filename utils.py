import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

#PLOT POLICY
def plot_policy(grid, Q, plot_name, grid_idx):
    """
    Generates a visualization of the learned optimal policy.

    Args:
        grid (np.array): The environment (matrix)
        Q (np.array): Trained Q table
        plot_name (str): Name for saving the PNG file
        grid_idx (int): Grid index for a better understanding
    """

    #All possible actions to take
    action_arrows = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←' 
    }
    
    #Get the grid size and create the plot
    grid_size = grid.shape[0]
    fig, ax = plt.subplots(figsize=(8,8))

    #Plot the grid
    for i in range(grid_size):
        for j in range(grid_size):
            cell = grid[i, j]
            rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black')
            ax.add_patch(rect)

            if cell == 'N':
                rect.set_facecolor("#72B842")
            elif cell == 'M':
                rect.set_facecolor("#5EAAF6")
            elif cell == 'P':
                rect.set_facecolor('#397716')
            else:
                rect.set_facecolor('#000000')

            #Get the optimal policy
            state = i * grid_size + j
            best_action = np.argmax(Q[state])
            arrow = action_arrows[best_action]

            #Paint the arrows where to show the policy
            ax.text(j + 0.5, i + 0.5, arrow, ha='center', va='center', fontsize=18, color='red')

    #Extra settings for the plot
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.title("Optimal policy Grid " + str(grid_idx), fontsize=16)

    #Save the plot
    plot_path = os.path.join(save_dir, plot_name + ".png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Policy saved as {plot_path}")



#PLOT REWARDS/ALGORITHMS
def smooth(x, window):
    """Returns a smoother curve of the rewards"""
    return [np.mean(x[max(0, i-window):i+1]) for i in range(len(x))]


def plot_rewards_comparison(algos_results, grid_idx, window):
    """Generates and saves a plot for the comparison of the different algorithms and their rewards.
    
    Args:
        algos_results (dict): Dictionary containing the results of each algoritms.
            - Each key is the algorithm name
            - Each value is another dictionary where:
                - 'Q': Trained Q table for the algorithm
                - 'rewards': Episode reward

        grid_idx (int): Grid index for a better understanding
        
        window (int): Size of the moving average window used to smooth the reard curves.

    Saves:
        A '.png' file named 'rewards_comparison_grid{grid_idx}.png' showing
        smoothed reward curves for all algorithms on the given grid.

    Example:
        plot_rewards_comparison(results_algos, grid_idx=0, window=500)
        Plot saved as "rewards_comparison_grid0.png"
    """
    
    plt.figure(figsize=(10, 6))

    for algo_name, data in algos_results.items():
        plt.plot(smooth(data['rewards'], window), label=algo_name)
    plt.xlabel("Episode")
    plt.ylabel(f"Total reward (last average {window})")
    plt.title(f"Grid {grid_idx+1} - Comparison between algorithms")
    plt.legend()
    plt.grid(True)

    #Save the plot
    plot_path = os.path.join(save_dir, f"rewards_comparison_grid{grid_idx}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {plot_path}")



import numpy as np
import matplotlib.pyplot as plt
import time



def run_experiment(env, algorithm, episodes, alpha, gamma, epsilon, epsilon_decay=None, epsilon_min=None, seeds = None, max_steps_episode = None):
    """
    
    """
    results = {}
    seeds = seeds

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    variants = {
        "no_decay": {"epsilon_decay": None, "epsilon_min": None},
        "with_decay": {"epsilon_decay": epsilon_decay, "epsilon_min": epsilon_min}
    }

    for grid_idx in range(len(env.grid_list)):
        env.set_grid(grid_idx)
        results[grid_idx] = {}
        print(f"\n===== GRID {grid_idx} =====")

        for variant, params in variants.items():
            print(f"Running variant: {variant}")
            rewards_all = []
            lengths_all = []
            Q_all = []
            start_time = time.time()

            for seed in seeds:
                Q, rewards, info = algorithm(
                    env=env,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    episodes=episodes,
                    epsilon_decay=params["epsilon_decay"],
                    epsilon_min=params["epsilon_min"],
                    seed=seed,
                    max_steps_episode = max_steps_episode
                )
                rewards_all.append(rewards)
                lengths_all.append(info["lengths"])
                Q_all.append(Q)

            elapsed = time.time() - start_time
            rewards_all = np.array(rewards_all)
            lengths_all = np.array(lengths_all)
            Q_all = np.array(Q_all)

            mean_rewards = np.mean(rewards_all, axis=0)
            std_rewards = np.std(rewards_all, axis=0)
            Q_avg = np.mean(Q_all, axis=0)


            # estadísticas finales (últimos 100 episodios)
            last100 = rewards_all[:, -100:]
            mean_last100 = np.mean(last100)
            std_last100 = np.std(last100)
            mean_len = np.mean(lengths_all[:, -100:])

            results[grid_idx][variant] = {
                "mean_rewards": mean_rewards,
                "std_rewards": std_rewards,
                "mean_last100": mean_last100,
                "std_last100": std_last100,
                "mean_length": mean_len,
                "time": elapsed,
                "Q": Q_avg
            }

            # === plot learning curve ===
            plt.figure(figsize=(8,4))
            plt.plot(mean_rewards, label=f"{variant}")
            plt.fill_between(
                np.arange(len(mean_rewards)),
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2
            )

            folder = f"plots/{algorithm.__name__}"
            os.makedirs(folder, exist_ok=True)  # crea la carpeta si no existe

            # Línea horizontal en y=0 para referencia
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')

            plt.title(f"Algorithm {algorithm.__name__} - Grid {grid_idx} - {variant}")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/{algorithm.__name__}/{algorithm.__name__}_grid{grid_idx}_{variant}.png")
            plt.close()


            output_file = f"plots/{algorithm.__name__}/results_summary_{algorithm.__name__}.txt"
            
            with open(output_file, "a") as f: 
                f.write(f"Grid {grid_idx} - {variant}: "
                f"mean_last100={mean_last100:.2f}, std={std_last100:.2f}, "
                f"mean_length={mean_len:.1f}, time={elapsed:.1f}s\n\n")
    return results




import optuna
def objective_optuna(trial, env, algo, episodes, seed, max_steps_episode):
    """
    Objetivo de Optuna: maximizar la media de recompensa de los últimos 100 episodios
    para un algoritmo dado, un grid específico y una semilla fija.
    """
    # Hiperparámetros a optimizar
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)
    use_decay = trial.suggest_categorical("use_decay", [True, False])

    if use_decay:
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.999)
        epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.1)
    else:
        epsilon_decay = None
        epsilon_min = None

    # Ejecutar el algoritmo con semilla fija y max_steps_episode fijo
    _, rewards, _ = algo(
        env=env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        episodes=episodes,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
        max_steps_episode=max_steps_episode
    )

    # Retornar la media de los últimos 100 episodios
    return np.mean(rewards[-100:])

def optimize_with_optuna(env, algos, episodes, seed, n_trials, max_steps_episode, n_jobs=1):
    """
    Devuelve los mejores hiperparámetros por algoritmo y grid, usando una semilla fija.
    n_jobs: número de núcleos a usar en paralelo (default 1, secuencial)
    """
    best_params_dict = {}

    for algo in algos:
        best_params_dict[algo.__name__] = {}
        for grid_idx in range(len(env.grid_list)):
            print(f"\n=== Optimizing {algo.__name__} on Grid {grid_idx} ===")
            env.set_grid(grid_idx)

            study = optuna.create_study(direction="maximize")
            
            # Usamos n_jobs para paralelizar trials
            study.optimize(
                lambda trial: objective_optuna(trial, env, algo, episodes, seed, max_steps_episode),
                n_trials=n_trials,
                n_jobs=n_jobs
            )

            best_params_dict[algo.__name__][grid_idx] = study.best_trial.params
            print(f"Best params for {algo.__name__}, Grid {grid_idx}: {study.best_trial.params}")

    return best_params_dict

import json
import os
# Guardar los mejores hiperparámetros
def save_best_params(best_params, filename="best_params.json"):
    with open(filename, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best params saved to {filename}")

# Cargar los hiperparámetros guardados
def load_best_params(filename="best_params.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            best_params = json.load(f)
        print(f"Loaded best params from {filename}")
        return best_params
    else:
        print(f"No file found at {filename}")
        return None
