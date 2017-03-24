import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCH = 5

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = dsets.ImageFolder('../data/imagenet/train', transform, download=True)
testData = dsets.ImageFolder('../data/imagenet/test', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

class VGG16(tnn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.layer1 = tnn.Sequential(

        # 1-1 conv layer
        tnn.Conv3d(3, 64, kernel_size=3, padding=1),
        tnn.BatchNorm3d(64),
        tnn.ReLU(),

        # 1-2 conv layer
        tnn.Conv3d(64, 64, kernel_size=3, padding=1),
        tnn.BatchNorm3d(64),
        tnn.ReLU(),

        # 1 Pooling layer
        tnn.MaxPool3d(kernel_size=2, stride=2))

    self.layer2 = tnn.Sequential(

        # 2-1 conv layer
        tnn.Conv3d(64, 128, kernel_size=3, padding=1),
        tnn.BatchNorm3d(128),
        tnn.ReLU(),

        # 2-2 conv layer
        tnn.Conv3d(128, 128, kernel_size=3, padding=1),
        tnn.BatchNorm3d(128),
        tnn.ReLU(),

        # 2 Pooling lyaer
        tnn.MaxPool3d(kernel_size=2, stride=2))

    self.layer3 = tnn.Sequential(

        # 3-1 conv layer
        tnn.Conv3d(128, 256, kernel_size=3, padding=1),
        tnn.BatchNorm3d(256),
        tnn.ReLU(),

        # 3-2 conv layer
        tnn.Conv3d(256, 256, kernel_size=3, padding=1),
        tnn.BatchNorm3d(256),
        tnn.ReLU(),

        # 3 Pooling layer
        tnn.MaxPool3d(kernel_size=2, stride=2))

    self.layer4 = tnn.Sequential(

        # 4-1 conv layer
        tnn.Conv3d(256, 512, kernel_size=3, padding=1),
        tnn.BatchNorm3d(512),
        tnn.ReLU(),

        # 4-2 conv layer
        tnn.Conv3d(512, 512, kernel_size=3, padding=1),
        tnn.BatchNorm3d(512),
        tnn.ReLU(),

        # 4 Pooling layer
        tnn.MaxPool3d(kernel_size=2, stride=2))

    self.layer5 = tnn.Sequential(

        # 5-1 conv layer
        tnn.Conv3d(512, 512, kernel_size=3, padding=1),
        tnn.BatchNorm3d(512),
        tnn.ReLU(),

        # 5-2 conv layer
        tnn.Conv3d(512, 512, kernel_size=3, padding=1),
        tnn.BatchNorm3d(512),
        tnn.ReLU(),

        # 5 Pooling layer
        tnn.MaxPool3d(kernel_size=2, stride=2))

    self.layer6 = tnn.Sequential(

        # 6 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        tnn.Linear(4096, 4096),
        tnn.BatchNorm1d(4096),
        tnn.ReLU())
        

    self.layer7 = tnn.Sequential(

        # 7 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        tnn.Linear(4096, 4096),
        tnn.BatchNorm1d(4096),
        tnn.ReLU())

    self.layer8 = tnn.Sequential(

        # 8 output layer
        tnn.Linear(4096, 1000),
        tnn.BatchNorm1d(1000),
        tnn.Softmax())

    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.layer5(out)
      vgg16_features = out.view(out.size(0), -1)
      out = self.layer6(vgg16_features)
      out = self.layer7(out)
      out = self.layer8(out)

      return vgg16_features, out

      
vgg16 = VGG16()
vgg16.cuda()

# Loss and Optimizer
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(EPOCH):
#  for i, (images, labels) in enumerate(trainLoader):
  for images, labels in trainLoader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = vgg16(images)
    loss = cost(outputs, labels)
    loss.backward()
    optimizer.step()

#    if (i+1) % 100 == 0 :
#      print ('Epoch [%d/%d], Iter[%d/%d] Loss. %.4f' %
#          (epoch+1, EPOCH, i+1, len(trainData)//BATCH, loss.data[0]))

# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
  images = Variable(images).cuda()
  outputs = vgg16(images)
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

