import os.path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import StandardClassifier
from data_loader import get_test_loader


cifar10_root = './cifar10'
gpu = 3
batch_size = 1

# data preparation
normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# testloader = get_test_loader(cifar10_root, batch_size, pin_memory=True)
image_dataset = torchvision.datasets.ImageFolder('/home/jwang/cycada_feature/cyclegan/scripts/results/cifar2carla/test_20/images/test',
                                                 train_transform)
testloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
class_names = image_dataset.classes
print('Test images: {}'.format(len(testloader.sampler)))
print('Classes are: {}'.format(class_names))

# model configuration
model = StandardClassifier(3, 10)
model.load_state_dict(torch.load(os.path.join(cifar10_root, 'snapshots/carla_style/cifar10_30.pth')))

model.eval()
model.cuda(gpu)

# test
with torch.no_grad():
    running_corrects = 0
    for batch in testloader:
        images, labels = batch
        images, labels = Variable(images).cuda(gpu), Variable(labels).cuda(gpu)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels)

accuracy = running_corrects.double() / len(testloader.dataset)
print(accuracy.item())