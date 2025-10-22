import numpy as np
import random

#Q-LEARNING
def q_learning(env, alpha, gamma, epsilon, episodes):
    Q_q = np.zeros((env.observation_space.n, env.action_space.n))
    q_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_q[state])
            next_state, reward, done, _, _ = env.step(action)
            best_next = np.argmax(Q_q[next_state])
            Q_q[state, action] += alpha * (reward + gamma * Q_q[next_state, best_next] - Q_q[state, action])
            state = next_state
            total_reward += reward
        q_rewards.append(total_reward)
    return Q_q, q_rewards


#MONTECARLO
def montecarlo(env, alpha, gamma, epsilon, episodes):  
    Q_mc = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {}
    mc_rewards = []

    for ep in range(episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            #Epsilon-greedy (exploration VS explotation)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_mc[state])
            
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
        
        #Q update (Every-visit MC)
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
    
    return Q_mc, mc_rewards
    


#SARSA
def SARSA(env, alpha, gamma, epsilon, episodes):
    Q_sarsa = np.zeros((env.observation_space.n, env.action_space.n))
    sarsa_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Acción inicial (epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_sarsa[state])
        
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Siguiente acción (epsilon-greedy)
            if random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q_sarsa[next_state])
            
            # Actualizar Q
            Q_sarsa[state, action] += alpha * (reward + gamma * Q_sarsa[next_state, next_action] - Q_sarsa[state, action])
            
            state, action = next_state, next_action
        
        sarsa_rewards.append(total_reward)
    
    return Q_sarsa, sarsa_rewards