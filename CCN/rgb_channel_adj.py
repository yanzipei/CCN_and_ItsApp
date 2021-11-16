import numpy as np
import torch

__all__ = ['get_rgb_channel_adj']

classes = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
classes.sort()

# red_channel_color_dict = {'Red': ['Yellow', 'Brown', 'Orange', 'Pink', 'Purple', 'White', 'Gray'],
#                           'Green': ['Blue', 'Black'],
#                           'Blue': ['Green', 'Black'],
#                           'Yellow': ['Red', 'Brown', 'Orange', 'Pink', 'Purple', 'White', 'Gray'],
#                           'Brown': ['Red', 'Yellow', 'Purple', 'Gray'],
#                           'Orange': ['Red', 'Yellow', 'Pink', 'Purple', 'White', 'Gray'],
#                           'Pink': ['Red', 'Yellow', 'Orange', 'Purple', 'White', 'Gray'],
#                           'Purple': ['Red', 'Yellow', 'Brown', 'Orange', 'Pink', 'White', 'Gray'],
#                           'White': ['Red', 'Yellow', 'Orange', 'Pink', 'Purple'],
#                           'Gray': ['Red', 'Yellow', 'Orange', 'Pink', 'Purple'],
#                           'Black': ['Green', 'Blue']}
#
# green_channel_color_dict = {'Red': ['Blue', 'Purple', 'Black'],
#                             'Green': ['Yellow', 'Brown', 'Orange', 'Pink', 'White', 'Gray'],
#                             'Blue': ['Red', 'Purple', 'Black'],
#                             'Yellow': ['Green', 'Brown', 'Orange', 'Pink', 'White', 'Gray'],
#                             'Brown': ['Green', 'Yellow', 'Gray'],
#                             'Orange': ['Green', 'Yellow', 'Pink', 'Gray'],
#                             'Pink': ['Green', 'Yellow', 'Orange', 'White', 'Gray'],
#                             'Purple': ['Red', 'Blue', 'Black'],
#                             'White': ['Green', 'Yellow', 'Pink'],
#                             'Gray': ['Green', 'Yellow', 'Brown', 'Orange', 'Pink'],
#                             'Black': ['Red', 'Blue', 'Purple']}
#
# blue_channel_color_dict = {'Red': ['Green', 'Yellow', 'Brown', 'Orange', 'Black'],
#                            'Green': ['Red', 'Yellow', 'Brown', 'Orange', 'Black'],
#                            'Blue': ['Pink', 'Purple', 'White', 'Gray'],
#                            'Yellow': ['Red', 'Green', 'Brown', 'Orange', 'Black'],
#                            'Brown': ['Red', 'Green', 'Yellow', 'Orange', 'Black'],
#                            'Orange': ['Red', 'Green', 'Yellow', 'Brown', 'Black'],
#                            'Pink': ['Blue', 'Purple', 'White'],
#                            'Purple': ['Blue', 'Pink', 'White', 'Gray'],
#                            'White': ['Blue', 'Pink', 'Purple'],
#                            'Gray': ['Blue', 'Purple'],
#                            'Black': ['Red', 'Green', 'Yellow', 'Brown', 'Orange']}

red_channel_color_dict = {'Red': ['Yellow', 'Brown', 'Orange', 'Pink', 'Purple', 'White'],
                          'Green': ['Blue', 'Black'],
                          'Blue': ['Green', 'Black'],
                          'Yellow': ['Red', 'Brown', 'Orange', 'Pink', 'Purple', 'White'],
                          'Brown': ['Red', 'Yellow', 'Purple'],
                          'Orange': ['Red', 'Yellow', 'Pink', 'Purple', 'White'],
                          'Pink': ['Red', 'Yellow', 'Orange', 'Purple', 'White'],
                          'Purple': ['Red', 'Yellow', 'Brown', 'Orange', 'Pink', 'White'],
                          'White': ['Red', 'Yellow', 'Orange', 'Pink', 'Purple'],
                          'Gray': [],
                          'Black': ['Green', 'Blue']}

green_channel_color_dict = {'Red': ['Blue', 'Purple', 'Black'],
                            'Green': ['Yellow', 'Brown', 'Orange', 'Pink', 'White'],
                            'Blue': ['Red', 'Purple', 'Black'],
                            'Yellow': ['Green', 'Brown', 'Orange', 'Pink', 'White'],
                            'Brown': ['Green', 'Yellow'],
                            'Orange': ['Green', 'Yellow', 'Pink'],
                            'Pink': ['Green', 'Yellow', 'Orange', 'White'],
                            'Purple': ['Red', 'Blue', 'Black'],
                            'White': ['Green', 'Yellow', 'Pink'],
                            'Gray': [],
                            'Black': ['Red', 'Blue', 'Purple']}

blue_channel_color_dict = {'Red': ['Green', 'Yellow', 'Brown', 'Orange', 'Black'],
                           'Green': ['Red', 'Yellow', 'Brown', 'Orange', 'Black'],
                           'Blue': ['Pink', 'Purple', 'White', ],
                           'Yellow': ['Red', 'Green', 'Brown', 'Orange', 'Black'],
                           'Brown': ['Red', 'Green', 'Yellow', 'Orange', 'Black'],
                           'Orange': ['Red', 'Green', 'Yellow', 'Brown', 'Black'],
                           'Pink': ['Blue', 'Purple', 'White'],
                           'Purple': ['Blue', 'Pink', 'White'],
                           'White': ['Blue', 'Pink', 'Purple'],
                           'Gray': [],
                           'Black': ['Red', 'Green', 'Yellow', 'Brown', 'Orange']}


def do_check(d):
    # check
    for k in d:
        colors = d[k]
        for c in colors:
            target = d[c]
            # assert k in target
            if k not in target:
                print(k, c)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def create_adjacency_matrix(classes, adjacent_color_dict):
    adj = torch.zeros([len(classes), len(classes)])
    for k in classes:
        a = adjacent_color_dict[k]
        i = classes.index(k)
        for c in a:
            j = classes.index(c)
            adj[i][j] = 1
    return adj


def get_rgb_channel_adj():
    red_channel_adj = create_adjacency_matrix(classes, red_channel_color_dict) + torch.eye(len(classes))
    green_channel_adj = create_adjacency_matrix(classes, green_channel_color_dict) + torch.eye(len(classes))
    blue_channel_adj = create_adjacency_matrix(classes, blue_channel_color_dict) + torch.eye(len(classes))

    rgb_channel_adj = torch.cat((torch.unsqueeze(red_channel_adj, 0),
                                 torch.unsqueeze(green_channel_adj, 0),
                                 torch.unsqueeze(blue_channel_adj, 0)), dim=0)

    assert rgb_channel_adj[0].ne(red_channel_adj).sum().item() == 0
    assert rgb_channel_adj[1].ne(green_channel_adj).sum().item() == 0
    assert rgb_channel_adj[2].ne(blue_channel_adj).sum().item() == 0

    return rgb_channel_adj


if __name__ == '__main__':
    do_check(red_channel_color_dict)
    do_check(green_channel_color_dict)
    do_check(blue_channel_color_dict)

    red_channel_adjacency_matrix = create_adjacency_matrix(classes, red_channel_color_dict) + torch.eye(len(classes))
    green_channel_adjacency_matrix = create_adjacency_matrix(classes, green_channel_color_dict) + torch.eye(
        len(classes))
    blue_channel_adjacency_matrix = create_adjacency_matrix(classes, blue_channel_color_dict) + torch.eye(len(classes))
    # print(f'color_adjacency_matrix:\n{color_adjacency_matrix}')
    print(f'red_channel_adjacency_matrix:\n{red_channel_adjacency_matrix}')
    print(f'green_channel_adjacency_matrix:\n{green_channel_adjacency_matrix}')
    print(f'blue_channel_adjacency_matrix:\n{blue_channel_adjacency_matrix}')

    rgb_channel_adjacency_matrix = torch.cat(
        (red_channel_adjacency_matrix, green_channel_adjacency_matrix, blue_channel_adjacency_matrix))
    rgb_channel_adjacency_matrix = rgb_channel_adjacency_matrix.reshape(-1, len(classes), len(classes))
    print(rgb_channel_adjacency_matrix.size())
    assert rgb_channel_adjacency_matrix[0].ne(red_channel_adjacency_matrix).sum().item() == 0
    assert rgb_channel_adjacency_matrix[1].ne(green_channel_adjacency_matrix).sum().item() == 0
    assert rgb_channel_adjacency_matrix[2].ne(blue_channel_adjacency_matrix).sum().item() == 0

    # print(check_symmetric(color_adjacency_matrix))

    for a in rgb_channel_adjacency_matrix:
        print(check_symmetric(a))

    print(rgb_channel_adjacency_matrix.shape)

    print(rgb_channel_adjacency_matrix)

    rgb_channel_adj = get_rgb_channel_adj()
    assert rgb_channel_adj.ne(rgb_channel_adjacency_matrix).sum().item() == 0
