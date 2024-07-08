import torch
import torchvision
import torchvision.transforms as transforms
import sys
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as torch_dist
import debugpy

from kn_util.dist import init_dist, is_main_process, all_gather
from kn_util.utils.debug import setup_debugpy, ddp_synchronize_trace, sync_decorator
from kn_util.data.collection_ops import nested_to

init_dist()
setup_debugpy()


def build_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def main():
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    trainloader, testloader = build_dataloaders()

    torch_dist.barrier()

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    batch_size = 4
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net = net.cuda()
    net = DistributedDataParallel(net)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            data = nested_to(data, "cuda")
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            losses = all_gather(loss)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


if __name__ == "__main__":
    main()
