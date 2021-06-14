import os
import numpy as np
import gym
import sapien_rl.env
import logging
import sys

class BC_policy():
    def __init__(self):
        # TODO: load your trained model
        self.method = "BC"
        
    def step(self, obs):
        # TODO
        action = np.zeros(8)
        return action

class SAC_policy():
    def __init__(self):
        # TODO: load your trained model
        self.method = "SAC"
        
    def step(self, obs):
        # TODO
        action = np.zeros(8)
        return action

class Heuristic_policy():
    def __init__(self):
        # TODO
        self.method = "Heuristic"
        
    def step(self, obs):
        # TODO
        action = np.zeros(8)
        return action

def eval(env, policy):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler("log_%s.txt" % policy.method)
    logger.addHandler(output_file_handler)
    success_cnt = 0
    for i in range(100):
        if policy.method == "BC":
            env.reset(level = 3)
            env.perturb_gripper()
            obs = env.get_obs()
        else:
            obs = env.reset(level = i)
        for t in range(150):
            obs, reward, done, info = env.step(policy.step(obs))
            if done:
                if info['eval_info']['success']:
                    success_cnt += 1
                    logger.info("Episode #%d: Success! %d steps!" % (i, t))
                else:
                    logger.info("Episode #%d: Failed to open the door!" % i)
                logger.info(str(info['eval_info']))
                break
    logger.info("success rate: %d%%" % success_cnt)

if __name__ == "__main__":
    env = gym.make('OpenCabinet_state_45267_link_0-v4')

    # Run the following code to evaluate your methods.

    #bc_policy = BC_policy()
    #eval(env, bc_policy)

    #sac_policy = SAC_policy()
    #eval(env, sac_policy)

    #heuristic_policy = Heuristic_policy()
    #eval(env, heuristic_policy)