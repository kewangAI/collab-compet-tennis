from unityagents import UnityEnvironment
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse

from collections import deque
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import os

def rolling_aver(scores):
    scores_deque = deque(maxlen=100)
    scores_100 = []
    for s in scores:
        scores_deque.append(s)
        scores_100.append(np.mean(scores_deque))

    return scores_100

def plot_save_score(scores, file_name):
    scores_100 = rolling_aver(scores)
    v_scores = np.array([range(1, len(scores)+1), scores, scores_100])
    np.savetxt(file_name, np.transpose(v_scores), delimiter=',')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig("training.pdf", bbox_inches='tight')


def train(env, model_path='model_dir', number_of_episodes = 50000, episode_length = 500):

    noise = 1.0
    noise_reduction = 1.0
    batchsize = 256

    model_dir = os.getcwd() + "/"+model_path
    model_files = glob.glob(model_dir+"/*.pt")
    for file in model_files:
        os.remove(file)
    os.makedirs(model_dir, exist_ok=True)

    buffer = ReplayBuffer(int(1e5))
    rewards_deque = deque(maxlen=100)
    rewards_total = []

    # initialize policy and critic
    maddpg = MADDPG()

    for episode in range(1, number_of_episodes+1):

        rewards_this_episode = np.asarray([0.0, 0.0])

        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations

        for episode_t in range(episode_length):

            actions = maddpg.act(obs, noise=noise)
            noise *= noise_reduction

            env_info = env.step(actions)[brain_name]

            next_obs = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # add data to buffer
            transition = (obs, actions, rewards, next_obs, dones)
            buffer.push(transition)

            rewards_this_episode += rewards

            obs = next_obs

            if any(dones):
                break


        # update once after every episode_per_update
        if len(buffer) > batchsize*4:
            for _ in range(4):
                for a_i in range(num_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        rewards_total.append(np.max(rewards_this_episode))
        rewards_deque.append(rewards_total[-1])
        average_score = np.mean(rewards_deque)

        print(episode, rewards_this_episode, rewards_total[-1], average_score)

        # saving model
        save_dict_list = []
        if episode % 1000 == 0 :
            for i in range(2):
                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list,
                           os.path.join(model_dir,
                          'episode-{}.pt'.format(episode)))

                torch.save(save_dict_list,'best.pt')
    return rewards_total

def test(env, model_file = 'best.pt', num_ep = 100):

    rewards_total = []
    dict_list = torch.load(model_file)
    
    maddpg = MADDPG()
    for i in range(2):
       maddpg.maddpg_agent[i].actor.load_state_dict(dict_list[i]['actor_params'])
       maddpg.maddpg_agent[i].actor_optimizer.load_state_dict(dict_list[i]['actor_optim_params'])
       maddpg.maddpg_agent[i].critic.load_state_dict(dict_list[i]['critic_params'])
       maddpg.maddpg_agent[i].critic_optimizer.load_state_dict(dict_list[i]['critic_optim_params'])


    for i in range(1, num_ep+1):  # play game for 100 episodes
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(2)  # initialize the score (for each agent)
        while True:
            actions = maddpg.act(states)  # select actions
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        rewards_total.append(np.max(scores))
        print('Scores from episode {}: {}'.format(i, scores))
    print('Average Score over {} episodes: {}'.format(num_ep, np.mean(rewards_total)))
    
if __name__=='__main__':


    parser = argparse.ArgumentParser()

    # Flow
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(train=True)
    parser.set_defaults(test=False)

    args = parser.parse_args()


    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    if args.train :
        scores =   train(env)
        ofile = "score_history.csv"
        plot_save_score(scores, ofile)

    if args.test:
        scores = test(env)



    env.close()
