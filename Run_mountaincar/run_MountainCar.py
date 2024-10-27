"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DQNPrioritizedReplay,DoubleDQN,DuelingDQN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
# from RL_brain1 import DoubleDQN

tf.disable_v2_behavior()


env = gym.make('MountainCar-v0',render_mode = "rgb_array")
env = env.unwrapped
env.reset(seed=21)
MEMORY_SIZE = 300

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )



with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=False,
    )


with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, output_graph=False)

with tf.variable_scope('Dueling_DQN'):
    dueling_DQN = DuelingDQN(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,e_greedy_increment=0.001,sess=sess, output_graph=False,dueling=True)

sess.run(tf.global_variables_initializer())

# fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或者使用 'X264' 来获得更好的压缩率
# out = cv2.VideoWriter('output7.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 120.0, (600, 400))  # 创建VideoWriter对象

def train(RL):

    total_steps = 0
    frame_counter = 0
    steps = []
    episodes = []
    for i_episode in range(30):
        # total_steps = 0
        observation = env.reset()
        observation = np.array(observation[0])  #新修改
        while True:
            # env.render()
            # if i_episode == 0:
            #     if frame_counter < 2000:
            #         print(frame_counter)
            #         img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            #         # print(img.shape)
            #         # out.write(img)
            #
            #         # 写入图像到视频
            #         # print(img.shape)
            #         out.write(img)
            #         frame_counter += 1



                # frame_counter += 1
                # if frame_counter >200:
                #     break

                # img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                # cv2.imshow("test", img)
                # cv2.waitKey(2)


            action = RL.choose_action(observation)

            observation_, reward, done, info,_ = env.step(action) #新修改

            if done: reward = 100

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))


his_dueling = train(dueling_DQN)

his_natural = train(RL_natural)

his_prio = train(RL_prio)


his_double= train(double_DQN)


# out.release()




# compare based on first success
# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')

# print(his_natural)
# # print(type(his_natural[1, 0:-1]))
# # print([0]+his_natural[1, 0:-1])
#





print("第一个结果")
his_natural1=np.insert(his_natural[1, 0:-1], 0, 0)
print(his_natural[1, :] - his_natural1)
# his_natural=his_natural[:,18:]
# his_natural1=np.insert(his_natural[1, 18:-1], 0, 0)
# print("第一个结果")
# print(his_natural[1, :] - his_natural1)
print(his_natural[1, :] - his_natural1)
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural1, c='b', label='Natural DQN')


print("第二个结果")
his_prio1=np.insert(his_prio[1, 0:-1], 0, 0)
print(his_prio[1, :] - his_prio1)
# his_prio = his_prio[:,18:]
# his_prio1=np.insert(his_prio[1, 0:-1], 0, 0)
# print("第二个结果")
# print(his_prio[1, :] - his_prio1)
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio1, c='r', label='Prioritized DQN')


print("第三个结果")
his_double1=np.insert(his_double[1, 0:-1],0,0)
print(his_double[1, :] - his_double1)
# his_double = his_double[:,18:]
# his_double1=np.insert(his_double[1, 0:-1],0,0)
plt.plot(his_double[0, :], his_double[1, :]- his_double1, c='g', label='Double DQN')


print("第四个结果")
his_dueling1=np.insert(his_dueling[1, 0:-1],0,0)
print(his_dueling[1, :] - his_dueling1)
# his_dueling = his_dueling[:,18:]
# his_dueling1=np.insert(his_dueling[1, 0:-1],0,0)
plt.plot(his_dueling[0, :], his_dueling[1, :]- his_dueling1, c='m', label='Dueling DQN')



plt.legend(loc='best')
plt.ylabel('Steps')
plt.xlabel('episode')
plt.grid()
plt.savefig('05.png', dpi=300)
plt.show()

# out.release()


