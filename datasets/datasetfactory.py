import torchvision.transforms as transforms
from datasets.omniglot import Omniglot

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all=all)

        else:
            print("Unsupported Dataset")
            assert False
