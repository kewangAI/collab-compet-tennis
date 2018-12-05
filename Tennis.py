from unityagents import UnityEnvironment
import numpy as np

from collections import deque
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch



def train(env, number_of_episodes = 30000, episode_length = 500):

    noise = 1.0
    noise_reduction = 1.0
    buffer = ReplayBuffer(int(1e5))
    batchsize = 256

    rewards_deque      = deque(maxlen=100)
    rewards_total        = []

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
        #if episode % 100 == 0 or episode == number_of_episodes - 1:
        #    avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward), np.mean(agent2_reward)]


        # saving model
        # save_dict_list = []
        # if save_info:
        #     for i in range(2):
        #         save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
        #                      'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
        #                      'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
        #                      'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
        #         save_dict_list.append(save_dict)
        #
        #         torch.save(save_dict_list,
        #                    os.path.join(model_dir, 'episode-{}.pt'.format(episode)))


def test():
    # for i in range(1, 6):  # play game for 5 episodes
    #     env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    #     states = env_info.vector_observations  # get the current state (for each agent)
    #     scores = np.zeros(num_agents)  # initialize the score (for each agent)
    #     while True:
    #         actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
    #         actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
    #         env_info = env.step(actions)[brain_name]  # send all actions to tne environment
    #         next_states = env_info.vector_observations  # get next state (for each agent)
    #         rewards = env_info.rewards  # get reward (for each agent)
    #         dones = env_info.local_done  # see if episode finished
    #         scores += env_info.rewards  # update the score (for each agent)
    #         states = next_states  # roll over states to next time step
    #         if np.any(dones):  # exit loop if episode finished
    #             break
    #     print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
    pass


if __name__=='__main__':

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

    train(env)

    env.close()