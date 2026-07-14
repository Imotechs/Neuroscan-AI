from torch.nn import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
transforms = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

train_ds = datasets.ImageFolder("datasets/train",transforms=transforms)
test_ds = datasets.ImageFolder("datasets/test",transforms=transforms)
va_ds = datasets.ImageFolder("datasets/val",transforms=transforms)

train_dl = DataLoader(train_ds,batch_size=32,shuffle=True)
test_dl = DataLoader(test_ds,batch_size=32,shuffle=True)
va_dl = DataLoader(va_ds,batch_size=32,shuffle=True)
classes = len(train_ds.classes)
print(classes)

