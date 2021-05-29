import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=320, mode='train'):
        """Initializes image paths."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        image_paths = []
        for name in (os.listdir(root)):
            one_image_path = os.path.join(root, name)
            image_paths += list(map(lambda x: os.path.join(one_image_path, x), os.listdir(one_image_path)))
        self.image_paths = image_paths
        self.image_size = image_size
        self.mode = mode


    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        G_path = image_path.replace(image_path.split('/')[-3], image_path.split('/')[-3] + '_GT')
        fn = image_path.split('/')[-1].split('-')[0]
        filename = fn + '-gt.png'
        GT_path = G_path.replace(G_path.split('/')[-1], filename)

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        Transform = []

        #1.Transforms
        Transform.append(T.CenterCrop(int(image.size[0] * 0.65)))
        Transform.append(T.Resize(self.image_size))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        OT = []
        OT.append(T.Resize(self.image_size))
        OT.append(T.ToTensor())
        OT = T.Compose(OT)
        oimage = OT(image)

        image = Transform(image)
        GT = Transform(GT)

        #2.Normalize
        Norm_ = T.Normalize(0.16,0.20)
        image = Norm_(image)

        return oimage, image, GT, fn

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=1, mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return data_loader
