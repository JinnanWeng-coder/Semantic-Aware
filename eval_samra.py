import argparse
import os
import random

import numpy as np
import scipy.io
import torch as T

import Environment_Platoon_SC as ENV
from ddpg_torch import Agent
from global_critic import Global_Critic


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained SAMRA-MARL policy.')
    parser.add_argument('--gap', type=float, default=25, help='Intra-platoon gap in meters.')
    parser.add_argument('--demand-bytes', type=int, default=6000, help='Semantic V2V payload before transform.')
    parser.add_argument('--u', type=int, default=20, help='Transform factor / semantic-symbol scaling.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--out-dir', default='Data/run_default', help='Directory for saved evaluation arrays.')
    parser.add_argument('--tag', default='SAMRA', help='Tag used in output filenames.')
    parser.add_argument('--checkpoint-dir', default='model/marl_ddpg/', help='Directory containing trained checkpoints.')
    parser.add_argument('--rollouts', type=int, default=200, help='Number of evaluation rollouts.')
    return parser.parse_args()


def script_relative_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def set_checkpoint_dir(module, checkpoint_dir):
    module.checkpoint_dir = os.path.abspath(checkpoint_dir)
    module.checkpoint_file = os.path.join(module.checkpoint_dir, module.name + '_ddpg')


def set_all_checkpoint_dirs(agents, global_agent, checkpoint_dir):
    checkpoint_dir = script_relative_path(checkpoint_dir)
    for agent in agents:
        for network in [agent.actor, agent.target_actor, agent.critic, agent.target_critic]:
            set_checkpoint_dir(network, checkpoint_dir)
    for network in [global_agent.global_critic1, global_agent.global_critic2,
                    global_agent.global_target_critic1, global_agent.global_target_critic2]:
        set_checkpoint_dir(network, checkpoint_dir)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    ENV.np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


def get_state(env, idx, size_platoon):
    v2i_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0
    v2v_abs = (
        env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60
    ) / 60.0

    v2i_fast = (
        env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[idx * size_platoon] + 10
    ) / 35

    v2v_fast = (
        env.V2V_channels_with_fastfading[
            idx * size_platoon,
            idx * size_platoon + (1 + np.arange(size_platoon - 1)),
            :,
        ]
        - env.V2V_channels_abs[
            idx * size_platoon,
            idx * size_platoon + (1 + np.arange(size_platoon - 1)),
        ].reshape(size_platoon - 1, 1)
        + 10
    ) / 35

    interference = (-env.Interference_all[idx] - 60) / 60
    v2v_load_remaining = np.asarray([env.V2V_demand_semantic[idx] / env.V2V_demand_size_semantic])

    return np.concatenate(
        (
            np.reshape(v2i_abs, -1),
            np.reshape(v2i_fast, -1),
            np.reshape(v2v_abs, -1),
            np.reshape(v2v_fast, -1),
            np.reshape(interference, -1),
            v2v_load_remaining,
        ),
        axis=0,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    scipy.io.loadmat('sem_table.mat')
    scipy.io.loadmat('VQA_table.mat')

    up_lanes = [i / 2.0 for i in
                [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2,
                 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in
                  [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2,
                   500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in
                  [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2,
                   866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in
                   [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2,
                    866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]]

    width = 750 / 2
    height = 1298 / 2
    size_platoon = 5
    n_veh = 20
    n_platoon = int(n_veh / size_platoon)
    n_RB = 4
    n_S = 2
    max_power = 30
    V2I_min = 540
    bandwidth = int(180000)
    V2V_size_semantic = int(args.demand_bytes * 8) / args.u

    env = ENV.Environ(
        down_lanes, up_lanes, left_lanes, right_lanes, width, height,
        n_veh, size_platoon, n_RB, V2I_min, bandwidth, V2V_size_semantic, args.gap
    )
    env.new_random_game()

    n_step_per_episode = int(env.time_slow / env.time_fast)
    marl_n_input = len(get_state(env=env, idx=0, size_platoon=size_platoon))
    marl_n_output = 4 + (size_platoon - 1)

    batch_size = 64
    gamma = 0.99
    alpha = 0.0001
    beta = 0.001
    update_actor_interval = 2
    noise = 0.2
    C_fc1_dims = 1024
    C_fc2_dims = 512
    C_fc3_dims = 256
    A_fc1_dims = 1024
    A_fc2_dims = 512
    tau = 0.005

    agents = []
    for index_agent in range(n_platoon):
        agent = Agent(
            alpha, beta, marl_n_input, tau, marl_n_output, gamma,
            C_fc1_dims, C_fc2_dims, C_fc3_dims, A_fc1_dims, A_fc2_dims,
            batch_size, n_platoon, index_agent, noise
        )
        agents.append(agent)

    global_agent = Global_Critic(
        beta, marl_n_input, tau, marl_n_output, gamma,
        C_fc1_dims, C_fc2_dims, C_fc3_dims, batch_size,
        n_platoon, update_actor_interval, noise
    )

    set_all_checkpoint_dirs(agents, global_agent, args.checkpoint_dir)
    global_agent.load_models()
    for agent in agents:
        agent.load_models()
        agent.noise = 0.0

    eval_reward = np.zeros(args.rollouts)
    eval_qoe = np.zeros(args.rollouts)
    eval_v2v_success = np.zeros(args.rollouts)
    eval_delay = np.zeros(args.rollouts)

    for i_episode in range(args.rollouts):
        transmission_completed = False
        record_global_reward = np.zeros(n_step_per_episode)
        record_qoe = np.zeros(n_step_per_episode)
        record_v2v_success = []

        env.V2V_demand_semantic = env.V2V_demand_size_semantic * np.ones(n_platoon, dtype=np.float16)
        env.individual_time_limit_semantic = env.time_slow * np.ones(n_platoon, dtype=np.float16)
        env.active_links_semantic = np.ones(int(env.n_Veh / env.size_platoon), dtype='bool')

        if i_episode % 100 == 0:
            env.renew_positions()
            env.renew_channel(n_veh, size_platoon)
            env.renew_channels_fastfading()

        state_old_all = []
        for i in range(n_platoon):
            state_old_all.append(get_state(env=env, idx=i, size_platoon=size_platoon))

        for i_step in range(n_step_per_episode):
            action_all_training = np.zeros([n_platoon, 4])
            v2v_length_action = np.zeros([n_platoon, size_platoon - 1], dtype=int)

            for i in range(n_platoon):
                action = agents[i].choose_action(np.asarray(state_old_all[i]))
                action = np.clip(action, -0.999, 0.999)

                action_all_training[i, 0] = ((action[0] + 1) / 2) * n_RB
                action_all_training[i, 1] = ((action[1] + 1) / 2) * n_S
                action_all_training[i, 2] = np.round(np.clip(((action[2] + 1) / 2) * max_power, 1, max_power))
                action_all_training[i, 3] = ((action[3] + 1) / 2)
                for j in range(size_platoon - 1):
                    v2v_length_action[i, j] = int(((action[n_platoon + j] + 1) / 2) * args.u)

            action_temp = action_all_training.copy()
            _, global_reward, _, _, _, _, _, v2v_success, qoe = env.act_for_training(
                action_temp, v2v_length_action
            )
            record_global_reward[i_step] = global_reward
            record_qoe[i_step] = qoe.copy()
            record_v2v_success.append(v2v_success)

            if not transmission_completed and np.all(env.V2V_demand_semantic <= 0):
                transmission_completed = True
                eval_delay[i_episode] = i_step + 1

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            state_old_all = [
                get_state(env=env, idx=i, size_platoon=size_platoon)
                for i in range(n_platoon)
            ]

        eval_reward[i_episode] = np.mean(record_global_reward)
        eval_qoe[i_episode] = np.mean(record_qoe)
        eval_v2v_success[i_episode] = np.mean(record_v2v_success)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f'eval_reward_{args.tag}_seed{args.seed}.npy'), eval_reward)
    np.save(os.path.join(args.out_dir, f'eval_qoe_{args.tag}_seed{args.seed}.npy'), eval_qoe)
    np.save(os.path.join(args.out_dir, f'eval_v2v_success_{args.tag}_seed{args.seed}.npy'), eval_v2v_success)
    np.save(os.path.join(args.out_dir, f'eval_delay_{args.tag}_seed{args.seed}.npy'), eval_delay)

    print(
        f'tag={args.tag} gap={args.gap} demand={args.demand_bytes} u={args.u} | '
        f'reward={np.mean(eval_reward):.3f}±{np.std(eval_reward):.3f} '
        f'qoe={np.mean(eval_qoe):.3f}±{np.std(eval_qoe):.3f} '
        f'srs={np.mean(eval_v2v_success):.3f}±{np.std(eval_v2v_success):.3f} '
        f'delay={np.mean(eval_delay):.3f}±{np.std(eval_delay):.3f}'
    )


if __name__ == '__main__':
    main()
