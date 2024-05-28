 import numpy as np
import pygame
from random import randint as r
import random

#define global variables
green = 1
red = -1
end = 1000
d = 0
n = 10
scrx = n*50
scry = n*50
background = (255,255,255)
screen = pygame.display.set_mode((scrx,scry))
colors = [(255,255,255) for i in range(n**2)]



reward = np.array([[0,-1,0,0,0,0,0,0,0,0],
                   [0,0,0,+5,0,-1,0,0,0,0],
                   [+5,0,0,0,0,-10,+5,0,0,0],
                   [-1,-1,0,-1,-1,-1,-1,0,0,0],
                   [0,+5,-1,0,-1,+5,-1,-1,-1,0],
                   [0,0,-1,0,-1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,+5,0,0,0,0,-1,-1,-1,-1],
                   [0,-1,-1,-1,-1,-1,0,0,0,0],
                   [0,0,0,0,0,0,0,-1,+5,+10]])

# CHANGING THE VALUES OF REWARDS AND DOING KIND OF A NORMALIZATION
for i in range(10):
    for j in range(10):
        if reward[i,j] == 5:
            reward[i,j] = 1
        if reward[i,j] == 0:
            reward[i,j] = -0.05

terminals = []
num_flags = 7


for i in range(n):
    for j in range(n):
        if reward[i][j] == red:
            colors[n*i+j] = (255,0,0)
        elif reward[i][j] == green:
            colors[n*i+j] = (0,255,0)


colors[n**2 - 1] = (0,255,255)
terminals.append(n**2 - 1)

Q = np.zeros((n**2,4))

actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3}
states = {}
F = []
flag_counter = 0
k = 0
for i in range(n):
    for j in range(n):
        states[(i,j)] = k
        if reward[i,j] == green:
            F.append(k)
        k+=1

alpha = 0.01
gamma = 0.9
current_pos = [0,0]

epsilon = 0.8

def select_action(current_state):
    global current_pos,epsilon
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[1] != 0:
            if reward[current_pos[0], current_pos[1]-1] != red:
                possible_actions.append("left")
        if current_pos[1] != n-1:
            if reward[current_pos[0], current_pos[1]+1] != red:
                possible_actions.append("right")
        if current_pos[0] != 0:
            if reward[current_pos[0]-1, current_pos[1]] != red:
                possible_actions.append("up")
        if current_pos[0] != n-1:
            if reward[current_pos[0]+1, current_pos[1]] != red:
                possible_actions.append("down")
        action = actions[possible_actions[r(0,len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0:
            if reward[current_pos[0]-1, current_pos[1]] != red:
                possible_actions.append(Q[current_state,0])
            else:
                possible_actions.append(m - 100)
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1:
            if reward[current_pos[0]+1, current_pos[1]] != red:
                possible_actions.append(Q[current_state,1])
            else:
                possible_actions.append(m - 100)
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0:
            if reward[current_pos[0], current_pos[1]-1] != red:
                possible_actions.append(Q[current_state,2])
            else:
                possible_actions.append(m - 100)
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n-1:
            if reward[current_pos[0], current_pos[1]+1] != red:
                possible_actions.append(Q[current_state,3])
            else:
                possible_actions.append(m - 100)
        else:
            possible_actions.append(m - 100)

        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)])
    return action

def layout():
    c = 0
    for i in range(0,scrx,50):
        for j in range(0,scry,50):
            pygame.draw.rect(screen,(0,0,0),(j,i,j+50,i+50),0)
            pygame.draw.rect(screen,colors[c],(j+5,i+5,j+45,i+45),0)
            c+=1


    
          
result_episode = []
run = True
current_pos = [0,0]
while run:
    screen.fill(background)
    layout()
    pygame.draw.circle(screen,(0,0,0),(current_pos[1]*50 + 25,current_pos[0]*50 + 25),15,0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    current_state = states[(current_pos[0],current_pos[1])]
    action = select_action(current_state)
    result_episode.append(current_state)
    if action == 0:
        current_pos[0] -= 1
    elif action == 1:
        current_pos[0] += 1
    elif action == 2:
        current_pos[1] -= 1
    elif action == 3:
        current_pos[1] += 1
    new_state = states[(current_pos[0],current_pos[1])]
    rr = 0
    if new_state not in terminals:
        if flag_counter < len(F):
            if new_state == F[flag_counter]:
                rr = reward[current_pos[0],current_pos[1]]
                print("new_state: ", new_state)
                reward[current_pos[0],current_pos[1]] = -0.05
                flag_counter += 1
            else:
                rr = 0
        Q[current_state,action] += alpha*(rr + gamma*(np.max(Q[new_state])) - Q[current_state,action])
        if reward[current_pos[0],current_pos[1]] == 1:
            reward[current_pos[0],current_pos[1]] = -0.05
    else:
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] - Q[current_state,action])
        print(result_episode)
        print('reahced the end! epsilon is: ', epsilon)
        print('the resulte episode is: ', result_episode)
        print('the Q-table up until now: ', Q)
       
        current_pos = [0,0]
        epsilon -= 1e-3
        print(terminals)
        print(F, flag_counter)
        reward = np.array([[0,-1,0,0,0,0,0,0,0,0],
                   [0,0,0,+5,0,-1,0,0,0,0],
                   [+5,0,0,0,0,-10,+5,0,0,0],
                   [-1,-1,0,-1,-1,-1,-1,0,0,0],
                   [0,+5,-1,0,-1,+5,-1,-1,-1,0],
                   [0,0,-1,0,-1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,+5,0,0,0,0,-1,-1,-1,-1],
                   [0,-1,-1,-1,-1,-1,0,0,0,0],
                   [0,0,0,0,0,0,0,-1,+5,+5]])
        for i in range(10):
            for j in range(10):
                if reward[i,j] == 5:
                    reward[i,j] = 1
                if reward[i,j] == 0:
                    reward[i,j] = -0.05
        flag_counter = 0
    

pygame.quit()
print (epsilon)
print(Q)
