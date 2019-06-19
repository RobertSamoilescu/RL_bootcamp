import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()

		self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=8, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
		self.bn3 = nn.BatchNorm2d(32)


		def conv2d_size_out(size, kernel_size=5, stride=2):
			return (size - (kernel_size - 1) - 1) // stride + 1

		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 4), 8), 4)
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 4), 8), 4)

		linear_input_size = convw * convh * 32
		self.l1 = nn.Linear(linear_input_size, 256)
		self.l2 = nn.Linear(256, outputs)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		
		x = x.view(x.size(0), -1)
		x = F.relu(self.l1(x))
		x = self.l2(x)

		return x


if __name__ == "__main__":
	img = torch.rand(1, 3, 40, 40)

	nn = DQN(img.size(2), img.size(3), 4)
	print(nn)
	
	out = nn.forward(img)
	print(out)
