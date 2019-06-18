import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cv2


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')

    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, _ = screen.shape
    
    screen = screen[int(screen_height*0.4):int(screen_height * 0.8), :]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, slice_range, :]
    
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = cv2.resize(screen, (90, 40))
    screen = np.transpose(screen, (2, 0, 1))

    return screen

if __name__ == "__main__":
    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    while True:
        get_screen(env)
        
        state, reward, done, info = env.step(env.action_space.sample())
        
        if done:
            break