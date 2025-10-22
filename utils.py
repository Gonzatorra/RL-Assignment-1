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

