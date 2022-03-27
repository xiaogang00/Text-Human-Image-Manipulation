import torch.utils.data as data
import os
import os.path
import random
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

import json
from nltk.tokenize import RegexpTokenizer
import pickle
from collections import defaultdict

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class Text_and_Pose_Dataset(data.Dataset):
    def __init__(self, data_root, data_json, TEXT_WORDS_NUM,
                 transform=None,
                 shuffle=False,
                 target_transform=None,
                 pickle_root='./datasets'):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.TEXT_WORDS_NUM = TEXT_WORDS_NUM

        self.shuffle = shuffle

        f = open(data_json)
        self.dict = json.load(fp=f)
        self.data_length = len(self.dict)
        self.root = data_root

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(pickle_root)
        self.data_json_name = data_json

    def load_captions(self, dict):
        all_captions = []
        for i in range(len(dict)):
            captions = dict[i]['captions']
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokens = self.tokenizer.tokenize(cap.lower())
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
        return all_captions

    def build_dictionary(self, captions):
        word_counts = defaultdict(float)
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        return [ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir):
        filepath = os.path.join(data_dir, 'captions.pickle')
        if not os.path.isfile(filepath):
            f = open('caption_all.json')
            dict_all = json.load(fp=f)
            captions_all = self.load_captions(dict_all)
            ixtoword, wordtoix, n_words = self.build_dictionary(captions_all)
            with open(filepath, 'wb') as f:
                pickle.dump([ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                ixtoword, wordtoix = x[0], x[1]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        return ixtoword, wordtoix, n_words

    def get_caption(self, caption_this_origin, img_name):
        rev = []

        caption_this_origin = caption_this_origin.replace("\ufffd\ufffd", " ")
        caption_this = self.tokenizer.tokenize(caption_this_origin.lower())
        for w in caption_this:
            if w in self.wordtoix:
                w = w.encode('ascii', 'ignore').decode('ascii')
                rev.append(self.wordtoix[w])

        sent_caption = np.asarray(rev).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.TEXT_WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.TEXT_WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))
            np.random.shuffle(ix)
            ix = ix[:self.TEXT_WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.TEXT_WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        dict_this = self.dict[index % self.data_length]
        img_name = dict_this['file_path']
        caption_this = dict_this['captions']
        caption_this_number = len(caption_this)

        if caption_this_number >= 2:
            sent_ix = random.randint(0, caption_this_number-1)
        else:
            sent_ix = 0
        caption_this_choose = caption_this[sent_ix]
        caps, cap_len = self.get_caption(caption_this_choose, img_name)

        caps = torch.Tensor(caps)

        crop_img, crop_pose = self.read_img_and_pose(img_name)

        img = Image.fromarray(crop_img)
        pose_img = Image.fromarray(crop_pose)
        if self.transform is not None:
            img = self.transform(img)
            pose_img = self.transform(pose_img)

        cap_len = [cap_len]
        cap_len = torch.Tensor(cap_len)
        return img, pose_img, caps.long(), cap_len.long(), img_name

    def readRGB(self, uri):
        img = cv2.imread(uri)
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        return img[:, :, [2, 1, 0]]

    def read_img_and_pose(self, impath):
        mask_img = self.readRGB(os.path.join(self.root.replace('/imgs', '/pose_imgs'), impath))
        img = self.readRGB(os.path.join(self.root, impath))

        mask_img_sum = np.sum(mask_img, axis=2)
        location = np.where(mask_img_sum > 0)
        if len(location[0]) == 0:
            h_min = 0
            h_max = mask_img_sum.shape[0]
        else:
            h_min = np.min(location[0])
            h_max = np.max(location[0])
        if len(location[1]) == 0:
            w_min = 0
            w_max = mask_img_sum.shape[1]
        else:
            w_min = np.min(location[1])
            w_max = np.max(location[1])

        h = img.shape[0]
        w = img.shape[1]
        h_min = max(0, h_min-5)
        w_min = max(0, w_min-5)
        h_max = min(h_max+5, h-1)
        w_max = min(w_max+5, w-1)

        crop_img = img[h_min:h_max, w_min:w_max, :]
        crop_pose = mask_img[h_min:h_max, w_min:w_max, :]
        return crop_img, crop_pose

    def __len__(self):
        return len(self.dict)



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
