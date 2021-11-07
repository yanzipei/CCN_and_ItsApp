from rgb_channel_adj import get_rgb_channel_adj

config = {
    'classes': ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'],
    'rgb_channel_adj': get_rgb_channel_adj(),
    'mean': [127.5000, 127.5000, 127.5000],
    'std': [73.9003, 73.9003, 73.9003],
    'mlp': {
        'hidden_dim': 18,
        'epochs': 10000,
        'optimizer': {
            'type': 'Adam',
            'lr': 0.01,
            'wd': 1e-06
        }
    },
    'color_mlp': {
        'hidden_dim': 18,
        'num_heads': 5,
        'alpha': 0.2,
        'l': [1e-05, 1e-05, 0.01],
        'epochs': 10000,
        'optimizer': {
            'type': 'Adam',
            'lr': 0.01,
            'wd': 1e-06
        }
    }
}

if __name__ == '__main__':
    # for k, v in config.items():
    #     print(f'{k}: {v}')
    import torch

    l = torch.tensor(config['color_mlp']['l']).unsqueeze(0)
    print(l.shape)
