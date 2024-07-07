from copy import copy

from train_loop import train

MAIN_EXPERIMENT_NAME = 'test' #Folder name of where tensorboard data is stored
NUM_RUNS = 1 #How many times to repeat experiment (helps when not setting seed)
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training loop.
    'alpha_scale' : .2,
    'alpha_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'policy_lr' : 3e-4,
    'sample_size' : 256,
    'tau' : .005,
    'seed' : 0
}

def train_hyper_param(name : str, values : list):
    '''Will train the hyperparam with the givin name for each value in values list.
    All possible parameter names can be found in train_loop's train function'''
    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for i in range(NUM_RUNS):
            try:
                train(f'{name}({value})_run({i})', log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment', **h_params)
            except ValueError: 
                with open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_run({i})/nan_v({value})_run({i}).txt', 'w', encoding='utf-8') as f:
                    f.write('I have NaNed')



if __name__ == '__main__':
    train_hyper_param('seed', [0, 1, 2])
