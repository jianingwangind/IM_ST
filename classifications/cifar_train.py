import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from models import StandardClassifier
from data_loader import get_train_valid_loader

cifar10_root = './cifar10'
num_epochs = 30
gpu = 2
batch_size = 128
augment = True
random_seed = 1999

# data preparation
normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
train_transform = transforms.Compose([
            # optional data augmentation
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

# dataloader = {}
# trainloader, validloader = get_train_valid_loader(cifar10_root, batch_size, augment, random_seed, pin_memory=True)
# dataloader['train'] = trainloader
# dataloader['valid'] = validloader

image_dataset = torchvision.datasets.ImageFolder('/home/jwang/cycada_feature/cyclegan/scripts/results/cifar2carla/test_20/images/train',
                                                 train_transform)
trainloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
class_names = image_dataset.classes

print('Training images: {}'.format(len(trainloader.sampler)))
print('Classes are: {}'.format(class_names))
# print('Validation images: {}'.format(len(validloader.sampler)))

# model configuration
model = StandardClassifier(3, 10)
model.cuda(gpu)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

# training without validation
for epoch in range(1, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    running_loss = 0.0
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()

        images, labels = batch
        images, labels = Variable(images).cuda(gpu), Variable(labels).cuda(gpu)
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('loss: %.3f' % (running_loss / len(trainloader)))

    if epoch % 5 == 0 or epoch == (num_epochs):
        print('Taking snapshot after epoch {}...'.format(epoch))
        torch.save(model.state_dict(), './cifar10/snapshots/carla_style/cifar10_{}.pth'.format(epoch))

    print('-' * 10)

# # training with validation
# for epoch in range(1, num_epochs + 1):
#     print('Epoch {}/{}'.format(epoch, num_epochs))
#     print('-' * 10)
#
#     running_loss = 0.0
#     running_corrects = 0
#     for phase in ['train', 'valid']:
#         if phase == 'train':
#             model.train()
#         elif phase == 'valid':
#             model.eval()
#
#         for i, batch in enumerate(dataloader[phase]):
#             optimizer.zero_grad()
#
#             images, labels = batch
#             images, labels = Variable(images).cuda(gpu), Variable(labels).cuda(gpu)
#             outputs = model(images)
#
#             if phase == 'train':
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#             elif phase == 'valid':
#                 with torch.no_grad():
#                     _, preds = torch.max(outputs, 1)
#                     running_corrects += torch.sum(preds == labels)
#
#     print('loss: %.3f' % (running_loss / len(dataloader['train'])))
#     print('acc: %.3f' % (running_corrects.double() / len(dataloader['valid'].sampler)))
#
#     if epoch % 5 == 0 or epoch == (num_epochs):
#         print('Taking snapshot after epoch {}...'.format(epoch))
#         torch.save(model.state_dict(), './cifar10/snapshots/cifar10_{}.pth'.format(epoch))
#     print('-' * 10)

print('Done')

