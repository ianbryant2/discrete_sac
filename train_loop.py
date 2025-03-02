from random import randint
import random
import csv
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor

import gymnasium as gym

from discrete import NNAgent
from data_collection import MemoryBuffer, Transition

#Hyperparameters

LR = 3e-4
HIDDEN_LAYER_SIZE = 256
POLICY_LR = LR
CRITIC_LR = LR
ALPHA_LR = LR
DISCOUNT = .99
TAU = .005
ALPHA_SCALE = .75
TARGET_UPDATE = 1 
UPDATE_FREQUENCY = 1 
EXPLORE_STEPS = 0
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 256


MAX_STEPS = 6e5

def train(
        extra_info : str = '', *,
        log_dir : str = '',
        hidden_size : int = HIDDEN_LAYER_SIZE,
        policy_lr : float = POLICY_LR,
        critic_lr : float = CRITIC_LR,
        alpha_lr : float = ALPHA_LR,
        discount : float = DISCOUNT,
        tau : float = TAU,
        alpha_scale :float = ALPHA_SCALE,
        target_update : int = TARGET_UPDATE,
        update_frequency : int = UPDATE_FREQUENCY,
        explore_steps : int = EXPLORE_STEPS ,
        buffer_size : int = BUFFER_SIZE,
        sample_size : int = SAMPLE_SIZE,
        max_steps : int = MAX_STEPS,
        environment_name : str = 'LunarLander-v2',
        seed : int = None,
        gpu : bool = True,) -> None:
    """Will be the main training loop"""

    h_params_dict = deepcopy(locals())
    del h_params_dict['extra_info']
    del h_params_dict['log_dir']
    _csv_of_hparams(log_dir + '/' + extra_info, h_params_dict)

    buffer = MemoryBuffer(buffer_size, sample_size, random)

    writer = SummaryWriter(log_dir + '/' + extra_info)
    
    #"LunarLander-v2"
    #"CartPole-v1"
    #"MountainCar-v0"
    env = gym.make(environment_name)
    
    seed_dict = dict() #A hack to passing in random seeds or not
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        seed_dict['seed'] = seed
        random.seed(seed)

    if torch.cuda.is_available() and gpu:
        device = f'cuda:{torch.cuda.current_device()}'
    else:
        device = 'cpu'

    device_obj = torch.device(device)

    torch.set_default_device(device_obj)

    agent = NNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden_size,
        policy_lr,
        critic_lr,
        alpha_lr,
        discount,
        tau,
        alpha_scale,
        target_update,
        update_frequency,
        writer
    )

    agent.to(device_obj)

    steps = 0
    num_games = 1
    total_return = 0
    previous_episodic_reward = 0
    episodic_reward = 0

    def get_action(s : Tensor) -> Tensor:
        if explore_steps <= steps:
            return agent(s)
        else:
            return torch.tensor(randint(0, env.action_space.n-1))
        
    state = torch.tensor(env.reset(**seed_dict)[0], device=device_obj, dtype=torch.float32)

    try:
        while max_steps > steps:
            action = get_action(state).unsqueeze(dim = 0)
            next_state, reward, terminated, truncated, _ = env.step(action.clone().detach().cpu().item())
            done = terminated or truncated
            total_return += reward
            episodic_reward += reward
            next_state = torch.tensor(next_state, device=device_obj, dtype=torch.float32)
            trans = Transition(
                state,
                action,
                next_state,
                torch.tensor([reward], device=device_obj, dtype=torch.float32),
                torch.tensor([terminated], device=device_obj, dtype=torch.float32)
            )

            buffer.add_data(trans)
            
            if explore_steps <= steps:
                agent.update(buffer.sample(), steps)

            steps += 1

            if done:
                next_state = torch.tensor(env.reset(**seed_dict)[0], device=device_obj, dtype=torch.float32)
                num_games += 1
                previous_episodic_reward = episodic_reward
                episodic_reward = 0
            
            writer.add_scalar('Average return', total_return / num_games, steps)
            writer.add_scalar('Episodic rerurn', previous_episodic_reward, steps)
            state = next_state

    finally:
        agent.save_actor(extra_info)
        env.close()

def _csv_of_hparams(log_dir : str, h_params_dict : dict):
    """Creates a csv at the log dir with the given hyperparameters"""

    file = Path(f'{log_dir}/hparams.csv')
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w+', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in h_params_dict.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    for i in range(3):
        train(i)
