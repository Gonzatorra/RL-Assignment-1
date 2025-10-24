from D_A_N_I_env import DaniEnv
from utils import plot_rewards_comparison, plot_policy, run_experiment, optimize_with_optuna, save_best_params, load_best_params, plot_algorithm, read_summary_file, load_rewards, analyze_reward_sensitivity, summarize_sensitivity_results
from algorithms import q_learning, montecarlo, SARSA

#import matplotlib
#matplotlib.use('Agg')

def main():
    #-----------------------------------------------#
    #-------------DEFINE PARAMETERS-----------------#
    #-----------------------------------------------#
    env = DaniEnv()
    episodes = 1000
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min =  0.05
    seed = 100
    algos = [q_learning, montecarlo, SARSA]
    max_steps_episode = 100


    #-----------------------------------------------#
    #---------STUDY INDIVIDUALLY ALGORITHMS---------#
    #-----------------------------------------------#
    for algo in algos:
        print(f"\n=== Running exhaustive analysis for {algo.__name__} ===")
        run_experiment(
            env=env,
            algorithm=algo,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            seed=seed,
            max_steps_episode=max_steps_episode
        )
    
    #Read results from saved txt file
    all_results = {}
    for algo in algos:
        txt_data = read_summary_file(algo.__name__)

        #Turn a list into a dictionary to handle easier the data --> {grid_idx: {variant: {...}}}
        results_dict = {}
        for entry in txt_data:
            grid = entry['grid_idx']
            var = entry['variant']
            results_dict.setdefault(grid, {})[var] = entry

        #all_results with all the necessary information to paint the plots
        all_results[algo.__name__] = results_dict


    print("\nLoaded results from TXT:")
    for algo_name, grids in all_results.items():
        total_entries = sum(len(variants) for variants in grids.values())
        print(f"{algo_name}: {total_entries} registros")



    #Generate learning curve for each algorithm and grid (decay/no decay)
    for algo_name, grids in all_results.items():
        for grid_idx, variants in grids.items():

            for variant_name, metrics in variants.items():
                #Load the necessary information from .npz
                mean_rewards, std_rewards = load_rewards(algo_name, grid_idx, variant_name)
                
                plot_algorithm(
                    algorithm_name=algo_name,
                    grid_idx=grid_idx,
                    variant=variant_name,
                    mean_rewards=mean_rewards,
                    std_rewards=std_rewards
                )


    #-----------------------------------------------#
    #--------------------OPTUNA---------------------#
    #-----------------------------------------------#


    #Try to load best parameters
    best_params = load_best_params()

    #If they do not exist, execute optuna and save them
    if best_params is None:
        best_params = optimize_with_optuna(
            env=env,
            algos=algos,
            episodes=1000,
            seed=100,
            n_trials=20,
            max_steps_episode=200,
            n_jobs=4  # usa 4 núcleos de CPU en paralelo
        )
        save_best_params(best_params)


    #Show best parameters for each algorithm and grid
    print("\n=== Best hyperparameters by algorithm and grid ===")
    for algo_name, grids in best_params.items():
        for grid_idx, params in grids.items():
            print(f"{algo_name} - Grid {grid_idx}: {params}")
        print("")

    #Execute each algorithm and each grid with the selected best parameters
    for grid_idx in range(len(env.grid_list)):
        print(f"\n=== Running algorithms on Grid {grid_idx} with best hyperparameters ===")
        env.set_grid(grid_idx)

        results_algos = {}

        for algo in algos:
            params = best_params[algo.__name__][str(grid_idx)]
            print(f"Running {algo.__name__} with params: {params}")

            Q, rewards, _ = algo(
                env=env,
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=params['epsilon'],
                episodes=episodes,
                epsilon_decay=params.get('epsilon_decay'),
                epsilon_min=params.get('epsilon_min'),
                seed=100,
                max_steps_episode=max_steps_episode
            )

            results_algos[algo.__name__] = {'Q': Q, 'rewards': rewards}

            #Plot its policy in each grid (for each algorithm)
            plot_policy(
                grid=env.grid_list[grid_idx],
                Q=Q,
                plot_name=f"{algo.__name__}_grid{grid_idx}_policy",
                grid_idx=grid_idx
            )

        #Reward comparison between algorithms in each grid
        plot_rewards_comparison(results_algos, grid_idx=grid_idx, window=50)
        
    env.close()


    #Reward value tuning para los algoritmos sobre un grid específico
    results_df = analyze_reward_sensitivity(algos)
    summarize_sensitivity_results(results_df)

    

if __name__ == "__main__":
    main()
