import numpy as np
import random
#this is an attempt to code a qlearning algo from scratch no assist

gamma = 0.9
lr = 0.5
n_episodes = 100
epsilon = 0.01
q_matrix = np.zeros((4,4))
                    #L  R  U  D
transition_matrix = [[0, 1, 0, 2],
                     [0, 1, 1, 3],           
                     [2, 3, 0, 2],           
                     [2, 3, 1, 3]]

def run_iteration():
    truncated = False
    terminated = False
    state = 0
    reward = 0
    i=0
    while not truncated and not terminated:
        action = pick_action(epsilon, q_matrix, state)
        new_state = transition_matrix[state][action]
        if new_state == 3:
            reward = 1

        q_matrix[state, action] += lr*(reward + gamma * max(q_matrix[new_state, :]) - q_matrix[state,action])
        if i > 10:
            truncated = True
        else:
            i+=1
        if reward == 1:
            terminated = True
        state = new_state

def pick_action(epsilon, q_matrix, state):
    rand = random.random()
    if epsilon < random.random():
        action = int(random.random()*4)
        print(action)
    else:
        action = np.argmax(q_matrix[state, :])
    return action


for i in range(0, n_episodes):
    run_iteration()
    epsilon +=0.01
    print("done")
    print(q_matrix)