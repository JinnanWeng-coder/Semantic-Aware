import numpy as np
import time
import random
import math
import os
import scipy.io

np.random.seed(1234)

mat_data1 = scipy.io.loadmat('sem_table.mat')
mat_data2 = scipy.io.loadmat('VQA_table.mat')

# 加载 'new_data_i_need.csv' 文件
single_table_data = mat_data1['sem_table']
bi_table_data = mat_data2['VQA_table']
symbol_text = [2, 4, 6, 8, 10]
symbol_image = [394, 788, 1576, 2364, 3152]

w = 0.8
class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2 #GHz
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db

class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # center of the grids
        self.shadow_std = 8

    
    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:

    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, size_platoon, n_RB,
                 V2I_min, BW, V2V_SIZE_semantic, Gap):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []
        self.V2V_pathloss = []
        self.V2V_channels_abs = []

        self.V2I_min = V2I_min #语义
        self.sig2_dB = -114 # dBm
        self.bsAntGain = 8 # 天线增益dBi
        self.bsNoiseFigure = 5 # dB
        self.vehAntGain = 3 # 天线增益dBi
        self.vehNoiseFigure = 9 # dB
        self.sig2 = 10 ** (self.sig2_dB / 10) # mW
        self.gap = Gap
        self.v_length = 0 #车身的长度忽略不计

        self.change_direction_prob = 0.4
        self.n_RB = n_RB
        self.n_Veh = n_veh
        self.size_platoon = size_platoon
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = BW  # bandwidth per RB, 180,000 MHz
        #self.V2V_demand_size = V2V_SIZE# V2V payload: 4000 Bytes every 100 ms
        self.V2V_demand_size_semantic = V2V_SIZE_semantic
        self.Interference_all = np.zeros(int(self.n_Veh / self.size_platoon)) + self.sig2

    def add_new_platoon(self, start_position, start_direction, start_velocity, size_platoon):
        for i in range(size_platoon):
            if start_direction == 'u':
                self.vehicles.append(Vehicle([start_position[0], start_position[1] - i * (self.gap + self.v_length)],
                                             start_direction, start_velocity))
            if start_direction == 'd':
                self.vehicles.append(Vehicle([start_position[0], start_position[1] + i * (self.gap + self.v_length)],
                                             start_direction, start_velocity))
            if start_direction == 'l':
                self.vehicles.append(Vehicle([start_position[0] + i*(self.gap + self.v_length), start_position[1]],
                                             start_direction, start_velocity))
            if start_direction == 'r':
                self.vehicles.append(Vehicle([start_position[0] - i*(self.gap + self.v_length), start_position[1]],
                                             start_direction, start_velocity))

    def add_new_platoon_by_number(self, number_vehicle, size_platoon):
        # due to the importance of initial positioning of platoons for RL, we have allocated their positions as follows:
        for i in range(int(number_vehicle / size_platoon)): #一共有这么多排
            # 对每个车队分配起始位置、方向和速度

            if i == 0:
                ind = 2
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)] # position of platoon leader
                self.add_new_platoon(start_position, 'd', np.random.randint(10, 15), size_platoon)
            elif i == 1:
                ind = 2
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]  # position of platoon leader
                self.add_new_platoon(start_position, 'u', np.random.randint(10, 15), size_platoon)
            elif i == 2:
                ind = 2
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'l', np.random.randint(10, 15), size_platoon)
            elif i == 3:
                ind = 2
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'r', np.random.randint(10, 15), size_platoon)
            elif i == 4:
                ind = 4
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)] # position of platoon leader
                self.add_new_platoon(start_position, 'd', np.random.randint(10, 15), size_platoon)
            elif i == 5:
                ind = 4
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]  # position of platoon leader
                self.add_new_platoon(start_position, 'u', np.random.randint(10, 15), size_platoon)
            elif i == 6:
                ind = 4
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'l', np.random.randint(10, 15), size_platoon)
            elif i == 7:
                ind = 4
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'r', np.random.randint(10, 15), size_platoon)

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each platoon
        # ===============
        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            # ================================================================================================
            if self.vehicles[i].direction == 'u':
                if i % self.size_platoon == 0:
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] <= self.left_lanes[j]) and \
                                ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.vehicles[i].position[0] -
                                                             (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                             self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] <= self.right_lanes[j]) and \
                                    ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] +
                                                                 (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                                 self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[1] += delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))  # vehicle i belongs to which platoon?
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[1] += delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'r':
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
                        else:
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
            # ================================================================================================
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] >= self.left_lanes[j]) and \
                                ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.vehicles[i].position[0] -
                                                             (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                             self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] >= self.right_lanes[j]) and \
                                    (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] +
                                                                 (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                                 self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[1] -= delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))  # vehicle i belongs to which platoon?
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[1] -= delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'r':
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
                        else:
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] + \
                                                           int(i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
            # ================================================================================================
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] <= self.up_lanes[j]) and \
                                ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] +
                                                             (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] <= self.down_lanes[j]) and \
                                    ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] -
                                                                 (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[0] += delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[0] += delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'u':
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
                        else:
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
            # ================================================================================================
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] >= self.up_lanes[j]) and \
                                ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] +
                                                             (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] >= self.down_lanes[j]) and \
                                    ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] -
                                                                 (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                        if change_direction == False:
                            self.vehicles[i].position[0] -= delta_distance

                else:
                    follow_index = int(np.floor(i / self.size_platoon))
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[0] -= delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'u':
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
                        else:
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
            # ================================================================================================
            # if it comes to an exit
            if i % self.size_platoon == 0:
                if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or \
                        (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                    if (self.vehicles[i].direction == 'u'):
                        self.vehicles[i].direction = 'r'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                    else:
                        if (self.vehicles[i].direction == 'd'):
                            self.vehicles[i].direction = 'l'
                            self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                        else:
                            if (self.vehicles[i].direction == 'l'):
                                self.vehicles[i].direction = 'u'
                                self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                            else:
                                if (self.vehicles[i].direction == 'r'):
                                    self.vehicles[i].direction = 'd'
                                    self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_channel(self, number_vehicle, size_platoon):
        """ Renew slow fading channel """
        # 路径损耗和阴影衰落

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = \
                    self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j],
                                                   self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = \
                    self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):

        """ Renew fast fading channel """
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))

    def Revenue_function(self, quantity, threshold):
        # G function definition in the paper
        revenue = 0
        if quantity >= threshold:
            revenue = 1
        else:
            revenue = 0
        return revenue

    def SINR_Scale1(self, arr):
        scale_factors = [-10, -5, 0, 5, 10, 15, 20]
        scaled_arr = []

        for num in arr:
            index = np.argmin(np.abs(num - np.array(scale_factors)))
            #scaled_value = scale_factors[index]
            scaled_arr.append(index)
        return scaled_arr

    def SINR_Scale2(self, arr):
        scale_factors = [-10, -5, 0, 5, 10, 15, 20]
        scaled_arr = np.empty(arr.shape, dtype=int)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                num = arr[i, j]
                index = np.argmin(np.abs(num - np.array(scale_factors)))
                #scaled_value = scale_factors[index]
                scaled_arr[i, j] = index
        return scaled_arr

    def Compute_Performance_Reward_Train(self, platoons_actions, V2V_length):
        sub_selection = platoons_actions[:, 0].astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)            # channel_selection_part
        platoon_decision = platoons_actions[:, 1].astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)         # platoon selection Intra/Inter platoon communication
        power_selection = platoons_actions[:, 2].reshape(int(self.n_Veh / self.size_platoon), 1)
        V2I_length = platoons_actions[:, 3].reshape(int(self.n_Veh / self.size_platoon), 1) #大小范围是0~1
        # ------------ Compute Interference --------------------
        self.platoon_V2I_Interference = np.zeros(int(self.n_Veh / self.size_platoon))  # V2I interferences
        self.platoon_V2I_Signal = np.zeros(int(self.n_Veh / self.size_platoon))  # V2I signals
        self.platoon_V2V_Interference = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon-1])  # V2V interferences
        self.platoon_V2V_Signal = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon-1])  # V2V signals

        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0: # platoon j has selected the inter-platoon communication
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #continue
                        self.platoon_V2I_Interference[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] - self.V2I_channels_with_fastfading[indexes[k, 0]*self.size_platoon, i] +
                                                       self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)  # mw
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1: # platoon j has selected the intra-platoon communication
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        for l in range(self.size_platoon-1):
                            self.platoon_V2V_Interference[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] - self.V2V_channels_with_fastfading[indexes[k, 0]*self.size_platoon, indexes[j, 0]*self.size_platoon + (l + 1), i] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # computing the platoons inter/intra-platoon signals
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                if platoon_decision[indexes[j, 0], 0] == 0:
                    self.platoon_V2I_Signal[indexes[j, 0]] = 10 ** ((power_selection[indexes[j, 0], 0] - self.V2I_channels_with_fastfading[indexes[j, 0]*self.size_platoon, i] +
                                                       self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                elif platoon_decision[indexes[j, 0], 0] == 1:
                    for l in range(self.size_platoon - 1):
                        self.platoon_V2V_Signal[indexes[j, 0], l] += 10 ** ((power_selection[indexes[j, 0], 0] - self.V2V_channels_with_fastfading[indexes[j, 0] * self.size_platoon, indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                    2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        V2I_SINR = np.divide(self.platoon_V2I_Signal, (self.platoon_V2I_Interference + self.sig2))
        V2V_SINR = np.divide(self.platoon_V2V_Signal, (self.platoon_V2V_Interference + self.sig2))
        V2V_SINR = np.clip(V2V_SINR, -10, 20)
        V2I_SINR = np.clip(V2I_SINR, -10, 20)
        V2I_SINR_index = self.SINR_Scale1(V2I_SINR)
        V2V_SINR_index = self.SINR_Scale2(V2V_SINR)
        # 没有语义
        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal, (self.platoon_V2I_Interference + self.sig2))) # 每秒每赫兹（bps/Hz）
        V2V_Rate = np.log2(1 + np.divide(self.platoon_V2V_Signal, (self.platoon_V2V_Interference + self.sig2))) # 每秒每赫兹（bps/Hz）

        V2I_semantic = np.zeros(int(self.n_Veh / self.size_platoon))
        V2I_number = int(self.n_Veh / self.size_platoon) - np.sum(platoon_decision)
        V2V_semantic = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon-1])
        V2I_Rate_semantic = np.zeros(int(self.n_Veh / self.size_platoon))
        V2V_Rate_semantic = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon - 1])

        # text transmission:
        # semantic emit rate is a random value from 50 to 70 in Ksuts/s;
        # beta follows CN(0.2,0.05^2) Gaussian distribution
        phi_S = 50 + (70 - 50) * np.random.rand(int(self.n_Veh / self.size_platoon), 1)  #
        beta_S = np.random.normal(0.2, 0.05, (int(self.n_Veh / self.size_platoon), 1))

        # image transmission:
        # semantic emit rate is a random value from 80 to 100 in Ksuts/s;
        # beta follows CN(0.1,0.02^2)
        phi_Bimage = 80 + (100 - 80) * np.random.rand(int(int(self.n_Veh / self.size_platoon)),
                                                      1)  # double semantic emission rate image transmission
        beta_Bimage = np.random.normal(0.1, 0.02, (int(int(self.n_Veh / self.size_platoon)), 1))

        # si is a random value from 0.8 to 0.9;
        # lamda follows CN(55,2.5^2)
        si = 0.8 + (0.9 - 0.8) * np.random.rand(int(self.n_Veh / self.size_platoon), 1)
        lamda = np.random.normal(55, 2.5, (int(self.n_Veh / self.size_platoon), 1))

        QoE = []
        if V2I_number == 0:
            for i in range(int(self.n_Veh / self.size_platoon)):
                if platoon_decision[i, 0] == 1:
                    V2V_semantic[i, 0] = single_table_data[V2V_length[i, 0], V2V_SINR_index[i, 0]]
                    V2V_semantic[i, 1] = single_table_data[V2V_length[i, 1], V2V_SINR_index[i, 1]]
                    V2V_semantic[i, 2] = single_table_data[V2V_length[i, 2], V2V_SINR_index[i, 2]]
                    V2V_semantic[i, 3] = single_table_data[V2V_length[i, 3], V2V_SINR_index[i, 3]]

                    V2V_Rate_semantic[i, 0] = 4 / ((V2V_length[i, 0]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 1] = 4 / ((V2V_length[i, 1]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 2] = 4 / ((V2V_length[i, 2]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 3] = 4 / ((V2V_length[i, 3]+1) / self.bandwidth)
                    QoE1=0
                    for j in range(self.size_platoon-1):
                        QoE1 += w / (1 + np.exp(beta_S[i] * (phi_S[i] - V2V_Rate_semantic[i, j] / 1000))) + (1 - w) / (1 + np.exp(lamda[i] * (si[i] - V2V_semantic[i, j])))
                    QoE.append(QoE1/4)

        if V2I_number == 1:#只有一个V2I链路，用单模态,其中20为单模态的句子长度
            for i in range(int(self.n_Veh / self.size_platoon)):
                if platoon_decision[i, 0] == 0:
                    V2I_semantic[i] = single_table_data[int(V2I_length[i, 0]*20), V2I_SINR_index[i]]
                    V2I_Rate_semantic[i] = 4 / (int(V2I_length[i, 0]*20 + 1) / self.bandwidth)

                    QoE1 = w / (1 + np.exp(beta_S[i] * (phi_S[i] - V2I_Rate_semantic[i] / 1000))) + (1 - w) / (
                                1 + np.exp(lamda[i] * (si[i] - V2I_semantic[i])))
                    QoE.append(QoE1)
                if platoon_decision[i, 0] == 1:
                    V2V_semantic[i, 0] = single_table_data[V2V_length[i, 0], V2V_SINR_index[i, 0]]
                    V2V_semantic[i, 1] = single_table_data[V2V_length[i, 1], V2V_SINR_index[i, 1]]
                    V2V_semantic[i, 2] = single_table_data[V2V_length[i, 2], V2V_SINR_index[i, 2]]
                    V2V_semantic[i, 3] = single_table_data[V2V_length[i, 3], V2V_SINR_index[i, 3]]

                    V2V_Rate_semantic[i, 0] = 4 / ((V2V_length[i, 0]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 1] = 4 / ((V2V_length[i, 1]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 2] = 4 / ((V2V_length[i, 2]+1) / self.bandwidth)
                    V2V_Rate_semantic[i, 3] = 4 / ((V2V_length[i, 3]+1) / self.bandwidth)
                    QoE2=0
                    for j in range(self.size_platoon-1):
                        QoE2 += w / (1 + np.exp(beta_S[i] * (phi_S[i] - V2V_Rate_semantic[i, j] / 1000))) + (1 - w) / (1 + np.exp(lamda[i] * (si[i] - V2V_semantic[i, j])))
                    QoE.append(QoE2 / 4)
        if V2I_number == 2:#有2个V2I链路, 其中5为多模态句子长度数组索引
            index = np.argwhere(platoon_decision == 0)
            V2I_list1 = bi_table_data[int(V2I_length[index[0, 0], 0]*5), int(V2I_length[index[1, 0], 0]*5)]
            V2I_semantic[index[0, 0]] = V2I_semantic[index[1, 0]] = V2I_list1[V2I_SINR_index[index[0, 0]], V2I_SINR_index[index[1, 0]]]

            V2I_Rate_semantic[index[0, 0]] = 2 / (symbol_text[int(V2I_length[index[0, 0], 0]*5)] / self.bandwidth)
            V2I_Rate_semantic[index[1, 0]] = (4 * 197) / (symbol_image[int(V2I_length[index[1, 0], 0]*5)] / self.bandwidth)

            QoE1 = w / (1 + np.exp(beta_S[index[0, 0]] * (phi_S[index[0, 0]] - V2I_Rate_semantic[index[0, 0]] / 1000))) + (1 - w) / (
                    1 + np.exp(lamda[index[0, 0]] * (si[index[0, 0]] - V2I_semantic[index[0, 0]])))
            QoE2 = w / (1 + np.exp(beta_Bimage[index[1, 0]] * (phi_Bimage[index[1, 0]] - V2I_Rate_semantic[index[1, 0]] / 1000))) + (1 - w) / (
                           1 + np.exp(lamda[index[1, 0]] * (si[index[1, 0]] - V2I_semantic[index[1, 0]])))
            QoE.append(QoE1)
            QoE.append(QoE2)
            for i in range(int(self.n_Veh / self.size_platoon)):
                if platoon_decision[i, 0] == 1:
                    V2V_semantic[i, 0] = single_table_data[V2V_length[i, 0], V2V_SINR_index[i, 0]]
                    V2V_semantic[i, 1] = single_table_data[V2V_length[i, 1], V2V_SINR_index[i, 1]]
                    V2V_semantic[i, 2] = single_table_data[V2V_length[i, 2], V2V_SINR_index[i, 2]]
                    V2V_semantic[i, 3] = single_table_data[V2V_length[i, 3], V2V_SINR_index[i, 3]]

                    V2V_Rate_semantic[i, 0] = 4 / ((V2V_length[i, 0] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 1] = 4 / ((V2V_length[i, 1] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 2] = 4 / ((V2V_length[i, 2] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 3] = 4 / ((V2V_length[i, 3] + 1) / self.bandwidth)
                    QoE3=0
                    for j in range(self.size_platoon-1):
                        QoE3 += w / (1 + np.exp(beta_S[i] * (phi_S[i] - V2V_Rate_semantic[i, j] / 1000))) + (1 - w) / (1 + np.exp(lamda[i] * (si[i] - V2V_semantic[i, j])))
                    QoE.append(QoE3/4)
        if V2I_number == 3:#有3个V2I链路
            index = np.argwhere(platoon_decision == 0)#寻找第几个车辆用V2I
            V2I_list1 = bi_table_data[int(V2I_length[index[0, 0], 0]*5), int(V2I_length[index[1, 0], 0]*5)]
            V2I_semantic[index[0, 0]] = V2I_semantic[index[1, 0]] = V2I_list1[V2I_SINR_index[index[0, 0]], V2I_SINR_index[index[1, 0]]]
            V2I_semantic[index[2, 0]] = single_table_data[int(V2I_length[index[2, 0], 0]*20), V2I_SINR_index[index[2, 0]]]

            V2I_Rate_semantic[index[0, 0]] = 2 / (symbol_text[int(V2I_length[index[0, 0], 0] * 5)] / self.bandwidth)
            V2I_Rate_semantic[index[1, 0]] = (4 * 197) / (symbol_image[int(V2I_length[index[1, 0], 0] * 5)] / self.bandwidth)
            V2I_Rate_semantic[index[2, 0]] = 4 / (int(V2I_length[index[2, 0], 0] * 20 +1) / self.bandwidth)

            QoE1 = w / (1 + np.exp(
                beta_S[index[0, 0]] * (phi_S[index[0, 0]] - V2I_Rate_semantic[index[0, 0]] / 1000))) + (1 - w) / (
                           1 + np.exp(lamda[index[0, 0]] * (si[index[0, 0]] - V2I_semantic[index[0, 0]])))
            QoE2 = w / (1 + np.exp(beta_Bimage[index[1, 0]] * (phi_Bimage[index[1, 0]] - V2I_Rate_semantic[index[1, 0]] / 1000))) + \
                   (1 - w) / (1 + np.exp(lamda[index[1, 0]] * (si[index[1, 0]] - V2I_semantic[index[1, 0]])))

            QoE3 = w / (1 + np.exp(beta_S[index[2, 0]] * (phi_S[index[2, 0]] - V2I_Rate_semantic[index[2, 0]] / 1000))) + \
                   (1 - w) / (1 + np.exp(lamda[index[2, 0]] * (si[index[2, 0]] - V2I_semantic[index[2, 0]])))
            QoE.append(QoE1)
            QoE.append(QoE2)
            QoE.append(QoE3)

            for i in range(int(self.n_Veh / self.size_platoon)):
                if platoon_decision[i, 0] == 1:
                    V2V_semantic[i, 0] = single_table_data[V2V_length[i, 0], V2V_SINR_index[i, 0]]
                    V2V_semantic[i, 1] = single_table_data[V2V_length[i, 1], V2V_SINR_index[i, 1]]
                    V2V_semantic[i, 2] = single_table_data[V2V_length[i, 2], V2V_SINR_index[i, 2]]
                    V2V_semantic[i, 3] = single_table_data[V2V_length[i, 3], V2V_SINR_index[i, 3]]

                    V2V_Rate_semantic[i, 0] = 4 / ((V2V_length[i, 0] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 1] = 4 / ((V2V_length[i, 1] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 2] = 4 / ((V2V_length[i, 2] + 1) / self.bandwidth)
                    V2V_Rate_semantic[i, 3] = 4 / ((V2V_length[i, 3] + 1) / self.bandwidth)
                    QoE4 = 0
                    for j in range(self.size_platoon-1):
                        QoE4 += w / (1 + np.exp(beta_S[i] * (phi_S[i] - V2V_Rate_semantic[i, j] / 1000))) + (1 - w) / (1 + np.exp(lamda[i] * (si[i] - V2V_semantic[i, j])))
                    QoE.append(QoE4/4)
        if V2I_number == 4:#有4个V2I链路, 其中5为多模态句子长度
            index = np.argwhere(platoon_decision == 0)  # 寻找第几个车辆用V2I
            V2I_list1 = bi_table_data[int(V2I_length[0, 0]*5), int(V2I_length[1, 0]*5)]
            V2I_semantic[0] = V2I_semantic[1] = V2I_list1[V2I_SINR_index[0], V2I_SINR_index[1]]
            V2I_list2 = bi_table_data[int(V2I_length[2, 0]*5), int(V2I_length[3, 0]*5)]
            V2I_semantic[2] = V2I_semantic[3] = V2I_list2[V2I_SINR_index[2], V2I_SINR_index[3]]

            V2I_Rate_semantic[0] = 2 / (symbol_text[int(V2I_length[0, 0] * 5)] / self.bandwidth)
            V2I_Rate_semantic[1] = (4 * 197) / (symbol_image[int(V2I_length[1, 0] * 5)] / self.bandwidth)
            V2I_Rate_semantic[2] = 2 / (symbol_text[int(V2I_length[2, 0] * 5)] / self.bandwidth)
            V2I_Rate_semantic[3] = (4 * 197) / (symbol_image[int(V2I_length[3, 0] * 5)] / self.bandwidth)

            QoE1 = w / (1 + np.exp(
                beta_S[0] * (phi_S[index[0, 0]] - V2I_Rate_semantic[index[0, 0]] / 1000))) + (1 - w) / (
                           1 + np.exp(lamda[index[0, 0]] * (si[index[0, 0]] - V2I_semantic[index[0, 0]])))
            QoE2 = w / (1 + np.exp(
                beta_Bimage[index[1, 0]] * (phi_Bimage[index[1, 0]] - V2I_Rate_semantic[index[1, 0]] / 1000))) + (
                           1 - w) / (
                           1 + np.exp(lamda[index[1, 0]] * (si[index[1, 0]] - V2I_semantic[index[1, 0]])))

            QoE3 = w / (1 + np.exp(
                beta_S[index[2, 0]] * (phi_S[index[2, 0]] - V2I_Rate_semantic[index[2, 0]] / 1000))) + (1 - w) / (
                           1 + np.exp(lamda[index[2, 0]] * (si[index[2, 0]] - V2I_semantic[index[2, 0]])))
            QoE4 = w / (1 + np.exp(
                beta_Bimage[index[3, 0]] * (phi_Bimage[index[3, 0]] - V2I_Rate_semantic[index[3, 0]] / 1000))) + (
                           1 - w) / (
                           1 + np.exp(lamda[index[3, 0]] * (si[index[3, 0]] - V2I_semantic[index[3, 0]])))
            QoE.append(QoE1)
            QoE.append(QoE2)
            QoE.append(QoE3)
            QoE.append(QoE4)
        # 没有语义
        #self.interplatoon_rate = V2I_Rate * self.time_fast * self.bandwidth
        #self.intraplatoon_rate = (V2V_Rate * self.time_fast * self.bandwidth).min(axis=1) # bits

        # # 有语义
        self.interplatoon_rate_semantic = V2I_Rate_semantic * self.time_fast
        self.intraplatoon_rate_semantic = (V2V_Rate_semantic * self.time_fast).min(axis=1)

        #没语义
        # self.V2I_demand -= self.interplatoon_rate
        #self.V2V_demand -= self.intraplatoon_rate
        # self.V2I_demand[self.V2I_demand < 0] = 0
        #self.V2V_demand[self.V2V_demand <= 0] = 0

        #self.individual_time_limit -= self.time_fast
        #self.active_links[np.multiply(self.active_links, self.V2V_demand <= 0)] = 0  # transmission finished, turned to "inactive"
        #reward_elements = self.intraplatoon_rate / 10000
        # reward_elements = np.zeros(int(self.n_Veh / self.size_platoon))
        #reward_elements[self.V2V_demand <= 0] = 1

        # # 有语义
        self.V2V_demand_semantic -= self.intraplatoon_rate_semantic
        self.V2V_demand_semantic[self.V2V_demand_semantic <= 0] = 0
        self.individual_time_limit_semantic -= self.time_fast
        self.active_links_semantic[np.multiply(self.active_links_semantic, self.V2V_demand_semantic <= 0)] = 0
        reward_elements_semantic = self.intraplatoon_rate_semantic / 500
        reward_elements_semantic[self.V2V_demand_semantic <= 0] = 1

       # return self.interplatoon_rate_semantic, self.intraplatoon_rate_semantic, self.V2V_demand_semantic, reward_elements
        return self.V2V_demand_semantic, reward_elements_semantic, V2I_semantic, V2V_semantic, self.interplatoon_rate_semantic, self.intraplatoon_rate_semantic, QoE

    def act_for_training(self, actions, V2V_length):
        per_user_reward = np.zeros(int(self.n_Veh / self.size_platoon))
        action_temp = actions.copy()
        V2V_demand_semantic, reward_elements_semantic,  V2I_semantic,\
        V2V_semantic, interplatoon_rate_semantic, intraplatoon_rate_semantic, QoE = self.Compute_Performance_Reward_Train(action_temp, V2V_length)

        V2V_success = 1 - np.sum(self.active_links_semantic) / (int(self.n_Veh / self.size_platoon))  # V2V success rates
        QoE = np.asarray(QoE, dtype=np.float64).reshape(-1)
        for i in range(int(self.n_Veh / self.size_platoon)):
            per_user_reward[i] = -(100) * (V2V_demand_semantic[i] / self.V2V_demand_size_semantic) + 10 * QoE[i]
        
        global_reward = np.mean(per_user_reward)
        return per_user_reward, global_reward, V2I_semantic, V2V_semantic, \
               interplatoon_rate_semantic, intraplatoon_rate_semantic, V2V_demand_semantic, V2V_success, np.sum(QoE)

    def act_for_testing(self, actions):
        action_temp = actions.copy()
        platoon_AoI, C_rate, V_rate, Demand, elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (int(self.n_Veh / self.size_platoon))  # V2V success rates

        return platoon_AoI, C_rate, V_rate, Demand, elements, V2V_success

    def Compute_Interference(self, platoons_actions):

        sub_selection = platoons_actions[:, 0].copy().astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)
        platoon_decision = platoons_actions[:, 1].copy().astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)
        power_selection = platoons_actions[:, 2].copy().reshape(int(self.n_Veh / self.size_platoon), 1)
        # ------------ Compute Interference --------------------
        V2I_Interference_state = np.zeros(int(self.n_Veh / self.size_platoon)) + self.sig2
        V2V_Interference_state = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon - 1]) + self.sig2

        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0:
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        V2I_Interference_state[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] - self.V2I_channels_with_fastfading[
                                indexes[k, 0] * self.size_platoon, i] +
                                    self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1:
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        for l in range(self.size_platoon - 1):
                            V2V_Interference_state[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] - self.V2V_channels_with_fastfading[
                                    indexes[k, 0] * self.size_platoon, indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        self.V2I_Interference_all = 10 * np.log10(V2I_Interference_state)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference_state)
        for i in range(int(self.n_Veh / self.size_platoon)):
            if platoon_decision[i, 0] == 0:
                self.Interference_all[i] = self.V2I_Interference_all[i]
            else:
                self.Interference_all[i] = np.max(self.V2V_Interference_all[i, :])
    def new_random_game(self, n_Veh=0):

        # make a new game
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_platoon_by_number(int(self.n_Veh), self.size_platoon)
        self.renew_channel(int(self.n_Veh), self.size_platoon)
        self.renew_channels_fastfading()

        '''self.V2V_demand = self.V2V_demand_size * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)
        self.individual_time_limit = self.time_slow * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)
        self.active_links = np.ones((int(self.n_Veh / self.size_platoon)), dtype='bool')
        self.AoI = np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)*100'''

        self.V2V_demand_semantic = self.V2V_demand_size_semantic * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16) / 20
        self.individual_time_limit_semantic = self.time_slow * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)
        self.active_links_semantic = np.ones((int(self.n_Veh / self.size_platoon)), dtype='bool')
        #self.AoI_semantic = np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16) * 100
