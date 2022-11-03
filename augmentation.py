import torchvision.transforms as transforms
import torch














class Augmentation():
    def __init__(self, normalize):
        if normalize.lower() == "imagenet":
            self.normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif normalize.lower() == "chestx-ray":
            self.normalize = transforms.Normalize(
                [0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        elif normalize.lower() == "none":
            self.normalize = None
        else:
            print(
                "mean and std for [{}] dataset do not exist!".format(normalize))
            exit(-1)

    def get_augmentation(self, augment_name, mode):
        try:
            aug = getattr(Augmentation, augment_name)
            return aug(self, mode)
        except:
            print("Augmentation [{}] does not exist!".format(augment_name))
            exit(-1)

    def basic(self, mode):
        transformList = []
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def _basic_crop(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
        else:
            transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_224(self, mode):
        transCrop = 224
        return self._basic_crop(transCrop, mode)

    def _basic_resize(self, size, mode="train"):
        transformList = []
        transformList.append(transforms.Resize(size))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_resize_224(self, mode):
        size = 224
        return self._basic_resize(size, mode)

    def _basic_crop_rot(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
            transformList.append(transforms.RandomRotation(7))
        else:
            transformList.append(transforms.CenterCrop(transCrop))

        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_rot_224(self, mode):
        transCrop = 224
        return self._basic_crop_rot(transCrop, mode)

    def _full(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "valid":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(transforms.Lambda(
                    lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        elif mode == "mytest":
            transResize = 280
            transCrop = 256

            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(transforms.Lambda(
                    lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_224(self, mode):
        transCrop = 224
        transResize = 256

        # transResize=224
        return self._full(transCrop, transResize, mode)

    def full_448(self, mode):
        transCrop = 448
        transResize = 512
        return self._full(transCrop, transResize, mode)

    def _full_colorjitter(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "valid":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(transforms.Lambda(
                    lambda crops: torch.stack([self.normalize(crop) for crop in crops])))

        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_colorjitter_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full_colorjitter(transCrop, transResize, mode)
