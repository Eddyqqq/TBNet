import os
import cv2
import sys
import pdb

import glob
import time
import torch

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from PIL import Image
import torch.utils.data as data

from utils import video_augmentation


sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, num_gloss=-1, mode="train", transform_mode=True):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./datainfo/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        input_data, label, ann, fi = self.read_process(idx)
        return input_data, torch.LongTensor(label), ann, self.inputs_list[idx]['original_info'].rsplit("|", 3)[0]

    def read_process(self, index):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        ann_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
                ann_list.append(phase)
        video = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
        video, label = self.data_aug(video, label_list, file_id=None)
        video = video.float() / 127.5 - 1
        return video, label, ann_list, fi

    def transform(self):
        if self.transform_mode == "train":
            print("training data transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
            ])
        else:
            print("testing data transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, ann, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, ann, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
