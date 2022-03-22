import gym
import gym_scm
import numpy as np
env = gym.make('scm-v0')

episodes=20

for e in range(episodes):
    done = False
    s = env.reset()
    while not done:
        print("state",s)
        a=env.action_space.sample()#select random action from action space
        print("action",a)
        s,r,done,info=env.step(env.action_space.sample())
        