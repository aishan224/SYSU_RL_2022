"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
"""
import argparse
from maze_env import Maze
from RL_q_learning import QLearning
from RL_sarsa import Sarsa

parser = argparse.ArgumentParser()
parser.add_argument('--flag', type=str, default='SARSA', choices=['SARSA', 'QLearning'])

def update( flag ):
    
    for episode in range(100):
        # initial observation
        observation = env.reset()

        if flag == 'SARSA':
            action = RL.choose_action(str(observation))

        while True:
            # fresh env
            '''Renders policy once on environment. Watch your agent play!'''
            env.render()

            '''
            RL choose action based on observation
            e.g: action = RL.choose_action(str(observation))
            '''
            
            ############################
            # YOUR IMPLEMENTATION HERE #
            if flag == 'QLearning':
                action = RL.choose_action(str(observation))
            ############################

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            if flag == 'SARSA':
                action_ = RL.choose_action(str(observation_)) # next action for SARSA

            '''
            RL learn from this transition
            e.g: RL.learn(str(observation), action, reward, str(observation_), is_lambda=True)
                 RL.learn(str(observation), action, reward, str(observation_), is_lambda_return=True)
            '''
            ############################

            # YOUR IMPLEMENTATION HERE #
            if flag == 'SARSA':
                RL.learn(str(observation), action, reward, str(observation_), action_)
            elif flag == 'QLearning':
                RL.learn(str(observation), action, reward, str(observation_))
            ############################

            # swap observation
            observation = observation_
            if flag == 'SARSA':
                action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()

    '''
    build RL Class
    RL = QLearning(actions=list(range(env.n_actions)))
    RL = Sarsa(actions=list(range(env.n_actions)))
    '''
    ############################

    # YOUR IMPLEMENTATION HERE #
    
    flag = parser.parse_args().flag
    # flag = 1 # 1: SARSA  0: QLearning
    if flag == 'SARSA':
        print("Run with SRASA")
        RL = Sarsa(actions=list(range(env.n_actions)))
    elif flag == 'QLearning':
        print("Run with QLearning")
        RL = QLearning(actions=list(range(env.n_actions)))

    ############################

    env.after(100, update(flag=flag))
    env.mainloop()

