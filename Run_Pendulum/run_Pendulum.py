"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN,DuelingDQN,DQNPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
import cv2
tf.disable_v2_behavior()



out = cv2.VideoWriter('output6.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 30.0, (500, 500))  # 创建VideoWriter对象

env = gym.make('Pendulum-v1',render_mode = "rgb_array")
# env = gym.make('Pendulum-v1',render_mode='rgb_array')
# env_render = gym.make('Pendulum-v1', render_mode='human')
env = env.unwrapped
# env_render = env_render.unwrapped
env.reset(seed=1)
# env_render.reset(seed=1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=False)


with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=False)


with tf.variable_scope('Priority_DQN'):
    Priority_DQN = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=False
    )

sess.run(tf.global_variables_initializer())

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train(RL):
    total_steps = 0
    observation = env.reset()
    observation = np.array(observation[0])

    frame_counter = 0  # 加在外面

    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()
        # env.render()   #新修改的
        # if total_steps>11000:


        if total_steps > 17000:
            # img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            # cv2.imshow("test", img)
            # cv2.waitKey(1)
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            # print(img.shape)
            # out.write(img)

            if frame_counter < 600:
                # 写入图像到视频
                print(img.shape)
                out.write(img)
                frame_counter += 1

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info,_= env.step(np.array([f_action]))

        # observation_, reward, done, info, _ = env.step(np.array([action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 16000:   # stop game
            # env.render()
            break

        observation = observation_
        total_steps += 1

    # #新增的测试部分
    # env.close()
    # observation = env_render.reset()
    # observation = np.array(observation[0])
    # for i in range(100000):
    #     action = RL.choose_action(observation)
    #
    #     f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
    #     observation_, reward, done, info, _ = env.step(np.array([f_action]))
    #
    #     # observation_, reward, done, info, _ = env.step(np.array([action]))
    #
    #     reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
    #     # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
    #     # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
    #
    #     RL.store_transition(observation, action, reward, observation_)
    #     env_render.render()
    #     if total_steps > MEMORY_SIZE:  # learning
    #         RL.learn()
    #     observation = observation_
    #     total_steps += 1
    return RL.q

#新增的

# out.release()




q_natural = train(natural_DQN)
# q_double = train(double_DQN)
# q_dueling=train(dueling_DQN)
# q_priority=train(Priority_DQN)


out.release()
# env.render()


# q_double = train(double_DQN)
# # env.render()
q_natural1=moving_average(q_natural, 9)
plt.plot(np.array(q_natural1), c='b', label='Natural DQN')

# q_double1=moving_average(q_double, 9)
# plt.plot(np.array(q_double1), c='g', label='Double DQN')
#
# q_dueling1=moving_average(q_dueling, 9)
# plt.plot(np.array(q_dueling1), c='m', label='Dueling DQN')
#
# q_priority1=moving_average(q_priority, 9)
# plt.plot(np.array(q_priority1), c='r', label='Prioritized DQN')



plt.legend(loc='best')
plt.ylabel('Q')
plt.xlabel('episode')
plt.grid()
plt.savefig('03.png', dpi=300)
plt.show()


# q_double1=moving_average(q_double,9)
#

# plt.plot(np.array(q_double1), c='b', label='double')
# plt.legend(loc='best')
# plt.ylabel('Q eval')
# plt.xlabel('training steps')
# plt.grid()
# plt.show()
