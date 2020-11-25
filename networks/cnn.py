import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN (nn.Module):
  def __init__(self, input_shape, batch_size=16, num_cats=50):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(128)
    self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(256)
    self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(256)
    self.dense1 = nn.Linear(176128,500)
    self.dropout = nn.Dropout(0.5)
    self.dense2 = nn.Linear(500, 2)
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = self.conv4(x)
    x = F.relu(self.bn4(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv5(x)
    x = F.relu(self.bn5(x))
    x = self.conv6(x)
    x = F.relu(self.bn6(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv7(x)
    x = F.relu(self.bn7(x))
    x = self.conv8(x)
    x = F.relu(self.bn8(x))
    x = x.view(x.size(0),-1)
    x = F.relu(self.dense1(x))
    x = self.dropout(x)
    x = self.dense2(x)
    return x
#   def __init__(self):
#       super(CNN, self).__init__()  # inherit all methods and properties of nn.Module ( parent )
#       self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
#       self.conv1_bn = nn.BatchNorm2d(64)

#       self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
#       self.conv2_bn = nn.BatchNorm2d(64)

#       self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
#       self.dropout0 = nn.Dropout(p=0.4)

#       # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
#       # self.conv3_bn = nn.BatchNorm2d(256)

#       self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1, 2])
#       self.conv4_bn = nn.BatchNorm2d(64)

#       self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

#       self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1, 2])
#       self.conv5_bn = nn.BatchNorm2d(64)

#       self.pool3 = nn.MaxPool2d(kernel_size=3, stride=[1, 2])

#       self.fc1 = nn.Linear(356640, 256)
#       self.dropout1 = nn.Dropout(p=0.3)
#       self.fc2 = nn.Linear(256, 32)
#       # self.dropout2 = nn.Dropout(p=0.3)
#       self.fc3 = nn.Linear(32, 1)

#   def forward(self, x):
#       # x = self.reshape_for_pytorch(x)
#       # x = x.permute(0, 3, 1, 2)
#       x = x.unsqueeze(1)
#       x = F.relu(self.conv1_bn(self.conv1(x)))
#       x = F.relu(self.conv2_bn(self.conv2(x)))
#       x = self.pool1(x)
#       x = self.dropout0(x)

#       # x = F.relu(self.conv3_bn(self.conv3(x)))
#       x = F.relu(self.conv4_bn(self.conv4(x)))
#       x = self.pool2(x)

#       x = F.relu(self.conv5_bn(self.conv5(x)))
#       x = self.pool3(x)

#       # Flattening to feed it to FFN
#       x = x.view(-1, x.shape[1:].numel())

#       #x = torch.cat([x, jitterx], dim=1)
#       x = F.relu(self.fc1(x))
#       x = self.dropout1(x)
#       x = F.relu(self.fc2(x))
#       # x = self.dropout2(x)
#       x = self.fc3(x)
#       return x

