import numpy as np
import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
	def __init__(self, capacity=10000):
		self.capacity = capacity
		self.size = 0

		self.buffer = []
		self.it = 0

	def append(self, elem):
		self.size = min(self.size + 1, self.capacity)

		# transform to tensors
		elem = [torch.tensor(x) for x in elem]

		# append element
		if self.size <= self.capacity:
			self.buffer.append(elem)
		else:
			self.buffer[self.it] = elem
			self.it = (self.it + 1) % self.capacity

	def sample(self, batch_size=128):
		return random.sample(self.buffer, batch_size)

	def __len__(self):
		return self.size


if __name__ == "__main__":
	buff = ReplayBuffer()

	buff.append(1)
	buff.append(2)

	print(buff.sample(2))
