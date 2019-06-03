# -*- coding: utf-8 -*-
from functools import lru_cache
from pathlib import Path

from PIL import Image
import pickle
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from ..layers.unet import UNetWithResnet50Encoder
from tqdm import tqdm
import os

from os.path import isfile, join
from os import listdir

class ImageFolderDataset(data.Dataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images not attached to any other
    label.

    This class also makes use of ``lru_cache`` to cache an image file once
    opened to avoid repetitive disk access.

    Arguments:
        root (str): The root folder that contains the images and index.txt
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.
        replicate(int, optional): Replicate the image names ``replicate``
            times in order to process the same image ``replicate`` times
            if ``replicate`` sentences are available during training time.
        warmup(bool, optional): If ``True``, the images will be read once
            at the beginning to fill the cache.
    """
    def __init__(self, root, resize=224, crop=224, replicate=1, warmup=False):
        index, raw_folder, layer, size = str(root).split("|")

        self.index = Path(index)
        self.raw = Path(raw_folder).expanduser().resolve()

        self.layer = layer
        self.size = int(size)
        self.feature_root = os.path.dirname(self.index)
        self.name = os.path.basename(os.path.splitext(self.index)[0])
        self.replicate = replicate


        _transforms = []
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        if not self.index.exists():
            raise(RuntimeError("index file not found {}".format(self.index)))

        self.raw_small = str(self.raw)+"_"+str(self.size)
        if not os.path.exists(self.raw_small):
            print("Folder", self.raw_small, "not found, extracting...")
            os.mkdir(self.raw_small)
            onlyfiles = [f for f in listdir(self.raw) if isfile(join(self.raw, f))]

            resize = self.size
            crop = self.size
            _transforms = []
            if resize is not None:
                _transforms.append(transforms.Resize(resize))
            if crop is not None:
                _transforms.append(transforms.CenterCrop(crop))
            transform = transforms.Compose(_transforms)

            for i in tqdm(range(len(onlyfiles))):
                o = onlyfiles[i]
                with open(os.path.join(self.raw, o), 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    x = transform(img)
                    x.save(os.path.join(self.raw_small, o), "JPEG", quality=100)


        features_filename = os.path.join(self.feature_root,"%s_unet_%s.pkl" %(self.name,str(size)))
        print("Loading %s" % features_filename)
        if not os.path.exists(features_filename):
            print(features_filename, "not found, extracting...")
            model = UNetWithResnet50Encoder(finetune=False).cuda()
            features = {}

            with open(self.index) as f:
                indexes = f.readlines()
                for i in tqdm(range(len(indexes))):
                    filename = os.path.join(self.raw_small, indexes[i].strip())
                    assert os.path.exists(filename), filename + " does not exists"
                    image = self._read_image(filename)
                    image_np16 = image.numpy().astype(np.float16)
                    inp = torch.from_numpy(image_np16).to(torch.float32)
                    inp = torch.unsqueeze(inp, 0).cuda()  # 1,dim,x,x
                    _, pre_pools = model.encode(inp)
                    for r in pre_pools.keys():
                        pre_pools[r] = pre_pools[r].squeeze(0).cpu().numpy().astype(np.float16)
                    features[i] = [pre_pools, image_np16]
                pickle.dump(features, open(features_filename, 'wb+'))

        self.features = pickle.load(open(features_filename, 'rb'))

        if warmup:
            for idx in range(self.__len__()):
                self[idx]

        # Replicate the list if requested

    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    @staticmethod
    def to_torch(batch):
        return torch.stack(batch)

    def __getitem__(self, idx):
        pre_pools, image = self.features[img_id]

        for r in pre_pools.keys():
            pre_pools[r] = pre_pools[r].astype(np.float32)
        image = image.astype(np.float32)

        return

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        s = "{}(replicate={}) ({} samples)\n".format(
            self.__class__.__name__, self.replicate, self.__len__())
        if self.transform:
            s += ' Transforms: {}\n'.format(
                self.transform.__repr__().replace('\n', '\n' + ' '))
        return s
