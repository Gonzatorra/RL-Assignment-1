#ALGORITMOS



# ---------------------
# Entrenamiento SECUENCIAL con reinicio de Q-table
# ---------------------
#env = FrozenLake8x8Env()
#
## Parámetros Q-learning
#alpha = 0.2
#gamma = 0.95
#epsilon = 1.0
#epsilon_min = 0.05
#epsilon_decay = 0.998
#episodes_per_grid = 6000  # Episodios por grid
#
## Almacenar resultados por grid
#grid_results = []
#
#print(f"Entrenando secuencialmente en {len(env.grid_list)} grids diferentes...")
#
## Para cada grid, entrenar desde cero
#for grid_idx in range(len(env.grid_list)):
#    print(f"\n=== COMENZANDO ENTRENAMIENTO EN GRID {grid_idx+1} ===")
#    
#    # Establecer el grid actual
#    env.set_grid(grid_idx)
#    
#    # REINICIAR la tabla Q para este grid (empezar desde cero)
#    Q = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
#    
#    # Reiniciar parámetros de exploración
#    epsilon = 1.0
#    rewards = []
#    success_count = 0
#    
#    # Entrenar en este grid específico
#    for ep in range(episodes_per_grid):
#        state, _ = env.reset()
#        done = False
#        total_reward = 0
#        steps = 0
#        
#        while not done and steps < 100:
#            # Epsilon-greedy
#            if random.uniform(0,1) < epsilon:
#                action = env.action_space.sample()
#            else:
#                action = np.argmax(Q[state])
#            
#            next_state, reward, done, _, _ = env.step(action)
#            
#            # Actualizar Q
#            best_next_action = np.argmax(Q[next_state])
#            td_target = reward + gamma * Q[next_state, best_next_action]
#            td_error = td_target - Q[state, action]
#            Q[state, action] += alpha * td_error
#            
#            state = next_state
#            total_reward += reward
#            steps += 1
#            
#            # Contar éxitos
#            if done and reward > 0:
#                success_count += 1
#        
#        # Decaimiento de epsilon
#        epsilon = max(epsilon_min, epsilon * epsilon_decay)
#        rewards.append(total_reward)
#        
#        # Mostrar progreso cada 500 episodios
#        if ep % 500 == 0:
#            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
#            success_rate = success_count / 500 if ep > 0 else 0
#            print(f"Grid {grid_idx+1}, Episodio {ep}: Recompensa promedio = {avg_reward:.2f}, Éxitos = {success_count}, Epsilon = {epsilon:.3f}")
#            success_count = 0
#    
#    # Guardar resultados de este grid
#    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
#    grid_results.append({
#        'grid_idx': grid_idx,
#        'final_avg_reward': final_avg,
#        'Q_table': Q.copy()  # Guardar la Q-table aprendida para este grid
#    })
#    
#    print(f"\n--- ENTRENAMIENTO COMPLETADO PARA GRID {grid_idx+1} ---")
#    print(f"Recompensa promedio últimos 100 episodios: {final_avg:.2f}")
#    
#    # Demostrar el agente en este grid inmediatamente después del entrenamiento
#    print(f"\nDemostración en Grid {grid_idx+1}:")
#    env.set_grid(grid_idx)  # Asegurarse de que estamos en el grid correcto
#    obs = env._get_state()
#    done = False
#    steps = 0
#    
#    while not done and steps < 50:
#        action = np.argmax(Q[obs])  # política greedy
#        obs, reward, done, _, _ = env.step(action)
#        env.render()
#        time.sleep(0.3)
#        steps += 1
#        
#        if done:
#            if reward > 0:
#                print(f"¡Éxito! Recompensa: {reward}")
#            else:
#                print(f"Fracaso. Recompensa: {reward}")
#            break
#    
#    if not done:
#        print(f"Tiempo agotado después de {steps} pasos")
#    
#    plot_policy(env, Q, "plot"+ str(grid_idx), grid_idx)
#
## Resumen final
#print("\n" + "="*50)
#print("RESUMEN FINAL DEL ENTRENAMIENTO")
#print("="*50)
#for result in grid_results:
#    print(f"Grid {result['grid_idx']+1}: Recompensa promedio = {result['final_avg_reward']:.2f}")
#
#env.close()


import matplotlib.pyplot as plt
import numpy as np
from D_A_N_I_env import DaniEnv
from utils import plot_rewards_comparison, plot_policy, run_experiment, optimize_with_optuna, save_best_params, load_best_params, plot_algorithm
from algorithms import q_learning, montecarlo, SARSA


def train_all_algorithms(env, grid_idx, episodes, alpha, gamma, epsilon):
    """
    Trains multiple reinforcement learning algorithms on a specific grid 
    environment and plots their optimal policies.

    Args:
        env (DaniEnv): The environment object where the agent will be trained.
        grid_idx (int): Index of the grid to train on.
        episodes (int): Number of episodes to train each algorithm.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for epsilon-greedy policies.

    Returns:
        dict: A dictionary containing the results for each algorithm. 
              Format:
              {
                  'Q-learning': {'Q': np.ndarray, 'rewards': list},
                  'Monte Carlo': {'Q': np.ndarray, 'rewards': list},
                  'SARSA': {'Q': np.ndarray, 'rewards': list}
              }

    """

    results = {}

    for algo_name, algo_func in [
        ('Q-learning', q_learning),
        ('Monte Carlo', montecarlo),
        ('SARSA', SARSA)
    ]:
        print(f"\n Training {algo_name} in Grid {grid_idx+1}...")
        Q, rewards = algo_func(env, alpha, gamma, epsilon, episodes)
        results[algo_name] = {'Q': Q, 'rewards': rewards}

        #Plot policy for the algorithm and grid (imported from utils.py)
        grid = env.grid_list[grid_idx]
        plot_policy(grid, Q, f"{algo_name}_grid{grid_idx}", grid_idx)

    return results

import matplotlib
matplotlib.use('Agg')

def main():
    env = DaniEnv()
    episodes = 1000
    alpha = 0.2
    gamma = 0.95
    epsilon, epsilon_decay, epsilon_min  = 1.0, 0.05, 0.998
    seed = [100, 101]
    algos = [q_learning, montecarlo, SARSA]
    max_steps_episode = 100
    n_trials = 30            # número de pruebas de Optuna


    #Experimentos exhaustivos -  ALGORITMOS INDIVIDUALES
    all_results = {}
    for algo in algos:
        print(f"\n=== Running exhaustive analysis for {algo.__name__} ===")
        results = run_experiment(
            env=env,
            algorithm=algo,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            seeds=seed,
            max_steps_episode=max_steps_episode
        )
        all_results[algo.__name__] = results

    # 📈 2️⃣ Generar curvas de aprendizaje
    for algo_name, grids in all_results.items():
        for grid_idx, variants in grids.items():
            for variant_name, metrics in variants.items():
                mean_rewards = metrics["mean_rewards"]
                std_rewards = metrics["std_rewards"]
                
                plot_algorithm(
                    algorithm_name=algo_name,
                    grid_idx=grid_idx,
                    variant=variant_name,
                    mean_rewards=mean_rewards,
                    std_rewards=std_rewards
                )


    # 1️⃣ Intentar cargar los mejores parámetros
    best_params = load_best_params()

    # 2️⃣ Si no existen, ejecutar Optuna y guardar
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

    # 3️⃣ Mostrar los mejores parámetros por algoritmo y grid
    print("\n=== Mejores hiperparámetros por algoritmo y grid ===")
    for algo_name, grids in best_params.items():
        for grid_idx, params in grids.items():
            print(f"{algo_name} - Grid {grid_idx}: {params}")

    # 2️⃣ Ejecutar cada algoritmo por grid usando los mejores parámetros
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

            # Plot policy para este algoritmo y grid
            plot_policy(
                grid=env.grid_list[grid_idx],
                Q=Q,
                plot_name=f"{algo.__name__}_grid{grid_idx}_policy",
                grid_idx=grid_idx
            )

        # Comparación de recompensas entre algoritmos
        plot_rewards_comparison(results_algos, grid_idx=grid_idx, window=50)

    env.close()


if __name__ == "__main__":
    main()
