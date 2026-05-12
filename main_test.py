import numpy as np
import os
import scipy.io
import Environment_Platoon_SC as ENV
from ddpg_torch import Agent
from buffer import ReplayBuffer
from global_critic import Global_Critic
import time
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

mat_data1 = scipy.io.loadmat('sem_table.mat')
mat_data2 = scipy.io.loadmat('VQA_table.mat')

# 加载 'new_data_i_need.csv' 文件
single_table_data = mat_data1['sem_table']
bi_table_data = mat_data2['VQA_table']


'''
---------------------------------------------------------------------------------------
Simulation code of the paper:
    "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent 
                        Multi-Task Reinforcement Learning"

Written by  : Mohammad Parvini, M.Sc. student at Tarbiat Modares University.
---------------------------------------------------------------------------------------
---> We have built our simulation following the urban case defined in Annex A of 
     3GPP, TS 36.885, "Study on LTE-based V2X Services".
---------------------------------------------------------------------------------------
'''
# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')
width = 750 / 2
height = 1298 / 2
IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
# ------------------------------------------------------------------------------------------------------------------ #
# simulation parameters:
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
size_platoon = 5 #一个排里有多少辆车
n_veh = 20  # n_platoon * size_platoon (需要改变)
n_platoon = int(n_veh / size_platoon)  # number of platoons
n_RB = 4  # number of resource blocks
n_S = 2  # decision parameter
Gap = 25 # meter 排内通信的间隔（需要改变）
symbol_length_text = [2, 4, 6, 8, 10]
symbol_length_image = [394, 788, 1576, 2364, 3152]

max_power = 30  # platoon leader maximum power in dbm ---> watt = 10^[(dbm - 30)/10]
V2I_min = 540 # minimum required data rate for V2I Communication = 3bps/Hz
bandwidth = int(180000)
V2V_size = int((4000) * 8) # V2V payload: 4000 Bytes every 100 ms

u= 20
V2I_min_semantic = 540
V2V_size_semantic = int((6000) * 8)/u

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, size_platoon, n_RB,
                  V2I_min, bandwidth, V2V_size_semantic, Gap)
env.new_random_game()  # initialize parameters in env

n_episode = 50
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
def get_state(env, idx):
    """ Get state from the environment """

    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0

    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60)/60.0

    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[
        idx * size_platoon] + 10) / 35

    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :]
                - env.V2V_channels_abs[idx * size_platoon, idx * size_platoon +
                                       (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35

    Interference = (-env.Interference_all[idx] - 60) / 60

    #AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))

    V2V_load_remaining = np.asarray([env.V2V_demand_semantic[idx] / env.V2V_demand_size_semantic])

    # time_remaining = np.asarray([env.individual_time_limit[idx] / env.time_slow])

    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), V2V_load_remaining), axis=0)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
marl_n_input = len(get_state(env=env, idx=0))
marl_n_output = 4+(size_platoon-1) # channel selection, mode selection, power ,length
# --------------------------------------------------------------
##---Initializations networks parameters---##
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
update_actor_interval = 2
noise = 0.2
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 1024
A_fc2_dims = 512
# ------------------------------

tau = 0.005
#--------------------------------------------
agents = []
for index_agent in range(n_platoon):
    print("Initializing agent", index_agent)
    agent = Agent(alpha, beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_platoon, index_agent, noise)
    agents.append(agent)
memory = ReplayBuffer(memory_size, marl_n_input, marl_n_output, n_platoon)
print("Initializing Global critic ...")
global_agent = Global_Critic(beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_platoon, update_actor_interval, noise)

global_agent.load_models()
for i in range(n_platoon):
    agents[i].load_models()
## Let's go
#AoI_evolution = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
Demand_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2I_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2V_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
power_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)

#AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_ = np.zeros([n_episode], dtype=np.float16)
record_QoE_ = np.zeros([n_episode], dtype=np.float16)
per_total_user_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_V2V_success_ = []

step_counter = 0  # 新增计数器
transmission_completed = False

if IS_TRAIN:
    # agent.load_models()
    record_critics_loss_ = np.zeros([n_platoon + 1, n_episode])
    record_global_reward_average = []
    for i_episode in range(n_episode):
        step_counter = 0  # 每个 episode 开始时重置计数器
        done = False
        transmission_completed = False  # 每个episode开始时重置标志
        print("----------------------------Episode: ", i_episode, "--------------------------------")
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)
        record_QoE = np.zeros([n_step_per_episode], dtype=np.float16)
        record_global_reward = np.zeros(n_step_per_episode)
        per_total_user = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
        record_V2V_success = []

        env.V2V_demand_semantic = env.V2V_demand_size_semantic * np.ones(n_platoon, dtype=np.float16)
        env.individual_time_limit_semantic = env.time_slow * np.ones(n_platoon, dtype=np.float16)
        env.active_links_semantic = np.ones((int(env.n_Veh / env.size_platoon)), dtype='bool')

        if i_episode % 100 == 0:
            env.renew_positions()  # update vehicle position
            env.renew_channel(n_veh, size_platoon)  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading

        state_old_all = []
        for i in range(n_platoon):
            state = get_state(env=env, idx=i)
            state_old_all.append(state)

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_platoon, 4])
            V2V_length_action = np.zeros([n_platoon, size_platoon-1], dtype=int)
            # receive observation
            for i in range(n_platoon):
                action = agents[i].choose_action(np.asarray(state_old_all[i]))
                action = np.clip(action, -0.999, 0.999)
                action_all.append(action)

                action_all_training[i, 0] = ((action[0]+1)/2) * n_RB  # chosen RB
                action_all_training[i, 1] = ((action[1]+1)/2) * n_S  # Inter/Intra platoon mode
                action_all_training[i, 2] = np.round(np.clip(((action[2]+1)/2) * max_power, 1, max_power))  # power selected by PL
                action_all_training[i, 3] = ((action[3]+1)/2)#因为不能先确定V2I链路数量，单模态还是多模态不确定，所以先给出0~1动作范围
                for j in range(size_platoon-1):
                    V2V_length_action[i, j] = int(((action[n_platoon+j]+1)/2) * 20)

            # All the agents take actions simultaneously, obtain reward, and update the environment
            action_temp = action_all_training.copy()
            training_reward, global_reward, V2I_semantic, V2V_semantic,\
            interplatoon_rate_semantic, intraplatoon_rate_semantic, V2V_demand_semantic, V2V_success, QoE= \
                env.act_for_training(action_temp, V2V_length_action)
            record_global_reward[i_step] = global_reward
            record_QoE[i_step] = QoE.copy()
            record_V2V_success.append(V2V_success)
            for i in range(n_platoon):
                per_total_user[i, i_step] = training_reward[i]

            # 检查传输是否完成
            if not transmission_completed and np.all(env.V2V_demand_semantic <= 0):
                transmission_completed = True  # 标记为已完成
                print(f"传输完成！当前 episode {i_episode} 的 step 数: {step_counter}")
                step_counter = 0  # 重置计数器（可选）
            else:
                step_counter += 1  # 未完成则计数器加1

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            # get new state
            for i in range(n_platoon):
                state_new = get_state(env, i)
                state_new_all.append(state_new)

            # old observation = new_observation
            for i in range(n_platoon):
                state_old_all[i] = state_new_all[i]

        for i in range(n_platoon):
            per_total_user_[i, i_episode] = np.mean(per_total_user[i])
            print('user', i, per_total_user_[i, i_episode], end='   ')

            '''for i in range(n_platoon):
                #AoI_evolution[i, i_episode % 100, i_step] = platoon_AoI[i]
                Demand_total[i, i_episode % 100, i_step] = V2V_demand_semantic[i]
                V2I_total[i, i_episode % 100, i_step] = interplatoon_rate_semantic[i]
                V2V_total[i, i_episode % 100, i_step] = intraplatoon_rate_semantic[i]
                power_total[i, i_episode % 100, i_step] = action_temp[i, 2]'''

        record_global_reward_average.append(np.mean(record_global_reward))
        record_QoE_[i_episode] = np.mean(record_QoE)
        record_V2V_success_.append(np.mean(record_V2V_success))
        print('agents rewards :', np.mean(record_global_reward), 'Sum QoE:', record_QoE_[i_episode], 'V2V success:',
              np.mean(record_V2V_success))
    print('average reward: ', np.mean(record_global_reward_average), 'average QoE:', np.mean(record_QoE_), 'V2V success:',
              np.mean(record_V2V_success_))
