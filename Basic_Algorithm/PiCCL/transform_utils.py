import random
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class BarlowTransform:
    def __init__(self, size=224, aggresive=False):
        self.aggresive = aggresive
        self.GaussianBlur_factor = 1 if size>50 else 0
        if self.aggresive != 0:
            print(f'Applying an Aggresive Transformation of order {self.aggresive}')
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=self.GaussianBlur_factor),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.6, contrast=0.6,
                                        saturation=0.6, hue=0.2)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=self.GaussianBlur_factor),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform3 = transforms.Compose([
            transforms.RandomResizedCrop(size, 
                                         ratio=(0.5, 2),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.6, contrast=0.6,
                                        saturation=0.4, hue=0.2)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=self.GaussianBlur_factor),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        if self.aggresive==1:
            y1 = self.transform2(x)
        elif self.aggresive==2:
            y1 = self.transform3(x)
        else:
            y1 = self.transform(x)
        #y2 = self.transform_prime(x)
        return y1#, y2

