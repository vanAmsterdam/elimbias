import random
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pandas as pd
import re
import numpy as np
import utils

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

def get_tfms_3d(split, params):
    if split == "train":
        def tfms(x):
            # x = random_3d_crop(x, params.n_crop_vox)
            
            x = normalize(x, params)
            x = random_crop(x, params.n_crop_vox)
            # batchgenerators transforms expect bath dim and channel dim
            # add these and squeeze off later
            x = np.expand_dims(np.expand_dims(x, 0), 0)
            # x = transforms3d.spatial_transforms.augment_mirroring(x)[0]
            x = np.squeeze(x)

            return x
    else:
        def tfms(x):
            x = normalize(x, params)
            x = unpad(x, int(params.n_crop_vox/2))
            return x
    
    return tfms

def get_tfms(split = "train", size = 51):
    if split == "train":
        tfms = transforms.Compose([
            # transforms.CenterCrop(70),
            transforms.RandomCrop(size),
            # transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            # transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(size, scale = (.9, 1)),
            # transforms.RandomRotation(12),
            # transforms.Resize((224, 224)),  # resize the image to 64x64 (remove if images are already 64x64),
            # transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            # transforms.RandomAffine(10, translate=(.1, .1), scale=(.1, .1), shear=.1, resample=False, fillcolor=0),
            transforms.ToTensor()
            # normalize_2d
            ])  # transform it into a torch tensor
    
    else:
        tfms = transforms.Compose([
            transforms.CenterCrop(size),
            # transforms.Resize((size, size)),
            transforms.ToTensor()
            # normalize_2d
            ])
    
    return tfms

def normalize_2d(x):
    return x / 255

def normalize(x, params=None):
    if params is None:
        MIN_BOUND = -1000.; MAX_BOUND = 600.0; PIXEL_MEAN = .25
    else:
        MIN_BOUND = params.hu_min; MAX_BOUND = params.hu_max; PIXEL_MEAN = params.pix_mean
    x = (x - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    x[x > (1 - PIXEL_MEAN)] = 1.
    x[x < (0 - PIXEL_MEAN)] - 0.
    return x

def random_crop(x, num_vox=3):
    starts = np.random.choice(range(num_vox), replace=True, size=(x.ndim,))
    ends = x.shape - (num_vox - starts)
    for i in range(x.ndim):
        x = x.take(indices=range(starts[i],ends[i]), axis=i)
    return x

def unpad(x, n=2):
    """
    Skim off n-entries in 3 dimensions
    """
    assert type(x) is np.ndarray
    if n>0:
        x = x[n:-n,n:-n,n:-n]
    return x

class LIDCDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform, df, setting, params):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.setting = setting
        self.params = params
        self.data_dir = data_dir
        self.transform = transform
        self.df = df
        self.mode3d = setting.mode3d
        self.covar_mode = setting.covar_mode
        self.fase = setting.fase

        # print(df.head())
        # print(df.dtypes)

        assert ("name" in df.columns)

        self.name_col  = df.columns.get_loc("name")
        self.label_col = df.columns.get_loc(setting.outcome[0])
        self.data_cols = list(set(range(len(self.df.columns))) - 
                                set([self.name_col, self.label_col]))

        # split of data, which contains covariate data that is not name or label
        if self.covar_mode:
            self.data = self.df.loc[:,"t"].values
        # if len(self.data_cols) > 0:
        #     self.data = self.df.iloc[:,self.data_cols]
        df['x_true'] = df.x

        # calculate a transformation of x to assess robustness of method to different measurements
        df['x'] = df.x + params.size_offset
        if params.size_measurement == 'area':
            pass
        elif params.size_measurement == 'diameter':
            df['x'] = df.x.values ** (1/2)
        elif params.size_measurement == 'volume':
            df['x'] = df.x.values ** (3/2)
        else:
            raise ValueError(f'dont know how to measure size in {params.size_measurement}, pick area, diameter or volume')
        # renormalize x to make sure that whatever measurement is used, the MSE is comparable
        df['x'] = (df.x - df.x.mean()) / df.x.std()


    def __len__(self):
        # return size of dataset
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        # image = Image.open(self.fpath_dict[self.idx_to_id[idx]]).convert("RGB")  # PIL image
        if self.mode3d:
            image = np.load(os.path.join(self.data_dir, self.df.iloc[idx, self.name_col]))
            image = image.astype(np.float32)
            image = self.transform(image)
            image = torch.from_numpy(image).unsqueeze(0)

        else:
            img_name = os.path.join(self.data_dir, 
                                    self.df.iloc[idx, self.name_col])
            image = Image.open(img_name).convert("L") # use rgb for resnet compatibility; L for grayscale
            image = self.transform(image)

        label = torch.from_numpy(np.array(self.df.iloc[idx, self.label_col], dtype = np.float32))

        sample = {"image": image, 'label': label}
        
        for variable in ["x", "y", "z", "t", 'x_true']:
            if variable in self.df.columns:
                sample[variable] = self.df[variable].values[idx].astype(np.float32)

        if self.setting.fase == "feature":
            sample[self.setting.outcome[0]] = self.df[self.setting.outcome[0]].values[idx].astype(np.float32)

        return sample

def fetch_dataloader(args, params, setting, types = ["train"], df = None):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        df: pandas dataframe containing at least name, label and split
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    if setting.gen_model == "":
        if setting.mode3d:
            data_dir = "data"
        else:
            data_dir = "slices"
    else:
        data_dir = os.path.join(setting.home, "data")

    if df is None:
        df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    dataloaders = {}

    if not setting.mode3d:
        pass
        # print(df.name.tolist()[:5])
        # df["name"] = df.apply(lambda x: os.path.join(x["split"], x["name"]), axis=1)
        # print(df.name.tolist()[:5])

    # make sure the dataframe has no index
    df_cols = df.columns
    df = df.reset_index()
    df = df[df_cols]

    try:
        assert setting.outcome[0] in df.columns
    except:
        print(f"outcome {setting.outcome[0]} not in df.columns:")
        print("\n".join(df.columns))
        raise


    if "split" in df.columns:
        splits = [x for x in types if x in df.split.unique().tolist()]
    else:
        df["split"] = types[0]
        splits = types

    df_grp = df.groupby("split")

    # for split in ['train', 'val', 'test']:
    for split, df_split in df_grp:
        df_split = df_split.drop("split", axis = 1)
        if split in types:
            # path = os.path.join(data_dir, split)
            path = data_dir
            if setting.mode3d:
                tfms = get_tfms_3d(split, params)
                # tfms = []
            else:
                tfms = get_tfms(split, params.size)

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(LIDCDataset(path, tfms, df_split, setting, params), 
                                        shuffle=True,
                                        num_workers=params.num_workers,
                                        batch_size=params.batch_size, 
                                        pin_memory=params.cuda)
                                        # batch_size = batch_size,
                                        # num_workers=2,
                                        # pin_memory=True)
            else:
                # dl = DataLoader(SEGMENTATIONDataset(path, eval_transformer, df[df.split.isin([split])]), 
                dl = DataLoader(LIDCDataset(path, tfms, df_split, setting, params), 
                                batch_size=params.batch_size,
                                num_workers=params.num_workers,
                                shuffle=False,
                                pin_memory=params.cuda)
                                # batch_size = batch_size,
                                # num_workers=2,
                                # pin_memory=True)

            dataloaders[split] = dl

    return dataloaders
