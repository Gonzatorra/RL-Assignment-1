import numpy as np
import random

#-----------------------------------------------#
#-------------------Q-LEARNING------------------#
#-----------------------------------------------#
def q_learning(env, alpha, gamma, epsilon, episodes, epsilon_decay=None, epsilon_min=None, seed=None, max_steps_episode=None):
    """
    Q-Learning algorithm with epsilon-greedy policy.

    Args:
        env (DaniEnv): Environment object
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate for epsilon-greedy
        episodes (int): Number of episodes to run
        epsilon_decay (float, optional): Factor to decay epsilon per episode
        epsilon_min (float, optional): Minimum allowed epsilon
        seed (int, optional): Random seed for reproducibility
        max_steps_episode (int, optional): Max steps allowed per episode

    Returns:
        Q_q (np.array): Learned Q-table
        q_rewards (list[float]): Total rewards per episode
        info (dict): Extra info (episode lengths)
    """

    #Set a seed if it is defined
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)

    
    Q_q = np.zeros((env.observation_space.n, env.action_space.n))
    q_rewards = []
    episode_lengths = []


    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0


        while not done and steps < max_steps_episode:

            #Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_q[state])
            
            next_state, reward, done, _, _ = env.step(action)
            best_next = np.argmax(Q_q[next_state])
            Q_q[state, action] += alpha * (reward + gamma * Q_q[next_state, best_next] - Q_q[state, action])
            
            state = next_state
            total_reward += reward
            steps +=1
        q_rewards.append(total_reward)
        episode_lengths.append(steps)

        #Apply epsilon decay if it is defined
        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    #Extra info with episodes lengths
    info = {"lengths": episode_lengths}
    return Q_q, q_rewards, info  





#-----------------------------------------------#
#-------------------MONTECARLO------------------#
#-----------------------------------------------#
def montecarlo(env, alpha, gamma, epsilon, episodes, epsilon_decay=None, epsilon_min=None, seed=None, max_steps_episode = None):  
    """
    Every-Visit Monte Carlo control with epsilon-greedy policy.

    Args:
        env (DaniEnv): Environment object
        alpha (float): Learning rate (not used for standard MC, included for compatibility)
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate for epsilon-greedy
        episodes (int): Number of episodes to run
        epsilon_decay (float, optional): Factor to decay epsilon per episode
        epsilon_min (float, optional): Minimum allowed epsilon
        seed (int, optional): Random seed for reproducibility
        max_steps_episode (int, optional): Max steps allowed per episode

    Returns:
        Q_mc (np.array): Learned Q-table
        mc_rewards (list[float]): Total rewards per episode
        info (dict): Extra info (episode lengths)
    """

    #Set a seed if it is defined
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)



    Q_mc = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {}
    mc_rewards = []
    episode_lengths = []

    #Generate complete episodes
    for ep in range(episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_episode: 
            #Epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample() #Exploration; with probability epsilon takes a random action.
            else:
                action = np.argmax(Q_mc[state]) #Explotation; with probability 1-epsilon takes de best action Q_mc.

            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            steps += 1

        #Every visit (takes into account all the actions)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            key = (state, action)
            if key not in returns:
                returns[key] = []
            returns[key].append(G)
            Q_mc[state, action] = np.mean(returns[key])

        mc_rewards.append(total_reward)
        episode_lengths.append(steps)

        #Apply epsilon decay if it is defined
        if epsilon_decay is not None and epsilon_min is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    #Extra info with episodes lengths
    info = {"lengths": episode_lengths}
    return Q_mc, mc_rewards, info
    





#-----------------------------------------------#
#---------------------SARSA---------------------#
#-----------------------------------------------#
def SARSA(env, alpha, gamma, epsilon, episodes, epsilon_decay=None, epsilon_min=None, seed=None, max_steps_episode = None):
    """
    SARSA algorithm (on-policy Temporal Difference control) with epsilon-greedy policy.

    Args:
        env (DaniEnv): Environment object
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate for epsilon-greedy
        episodes (int): Number of episodes to run
        epsilon_decay (float, optional): Factor to decay epsilon per episode
        epsilon_min (float, optional): Minimum allowed epsilon
        seed (int, optional): Random seed for reproducibility
        max_steps_episode (int, optional): Max steps allowed per episode

    Returns:
        Q_sarsa (np.array): Learned Q-table
        sarsa_rewards (list[float]): Total rewards per episode
        info (dict): Extra info (episode lengths)
    """
    
    #Set a seed if it is defined
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)

    
    Q_sarsa = np.zeros((env.observation_space.n, env.action_space.n))
    sarsa_rewards = []
    episode_lengths = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        #Acción inicial (epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_sarsa[state])
        

        while not done and steps < max_steps_episode:
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps +=1
            
            #Epsilon-greedy policy
            if random.random() < epsilon:
                next_action = env.action_space.sample() #Exploration; with probability epsilon takes a random action.
            else:
                next_action = np.argmax(Q_sarsa[next_state]) #Explotation; with probability 1-epsilon takes de best action Q_mc.
            
            #Update Q
            Q_sarsa[state, action] += alpha * (reward + gamma * Q_sarsa[next_state, next_action] - Q_sarsa[state, action])
            
            state, action = next_state, next_action

            
        sarsa_rewards.append(total_reward)
        episode_lengths.append(steps)


        #Apply epsilon decay if it is defined
        if epsilon_decay is not None and epsilon_min is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        
    #Extra info with episodes lengths
    info = {"lengths": episode_lengths}

    return Q_sarsa, sarsa_rewards, info