from utils_test import get_all_data_loaders_test, prepare_sub_folder, get_config
import argparse
from tester import Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import random
import numpy as np
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint_path', type=str, default='./pretrain', help="outputs path")
parser.add_argument('--output_path', type=str, default='./output', help="outputs path")
parser.add_argument('--test', action='store_true')
parser.add_argument('--name', type=str, default='test', help="outputs path")
opts = parser.parse_args()

cudnn.benchmark = True

config = get_config(opts.config)

test_loader_a, test_loader_b, ixtoword_test, wordtoix_test = get_all_data_loaders_test(config, shuffle1=False, shuffle2=True)
config['dataset_word_num'] = len(ixtoword_test)
trainer = Trainer(config)
trainer.cuda()

model_name = opts.name

checkpoint_directory = opts.checkpoint_path
image_directory = opts.output_path
if not(os.path.exists(image_directory)):
    os.mkdir(image_directory)


def prepare_data(data):
    img, pose_img, captions, captions_lens, label_list, aux_img, aux_pose_img, im_path = data
    captions_lens = captions_lens.squeeze(dim=1)
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    img = img[sorted_cap_indices]
    pose_img = pose_img[sorted_cap_indices]
    captions = captions[sorted_cap_indices]
    label_list = label_list[sorted_cap_indices]
    aux_img = aux_img[sorted_cap_indices]
    aux_pose_img = aux_pose_img[sorted_cap_indices]

    return [img, pose_img, captions, sorted_cap_lens, label_list, aux_img, aux_pose_img, im_path]


def writeTensor(save_path, tensor, nRow=16, row_first=False):
    nSample = tensor.shape[0]
    nCol = np.int16(nSample / nRow)
    all = []
    k = 0
    for iCol in range(nCol):
        all_ = []
        for iRow in range(nRow):
            now = tensor[k, :, :, :]
            now = now.permute(1, 2, 0)
            all_ += [now]
            k += 1
        if not row_first:
            all += [torch.cat(all_, dim=0)]
        else:
            all += [torch.cat(all_, dim=1)]
    if not row_first:
        all = torch.cat(all, dim=1)
    else:
        all = torch.cat(all, dim=0)
    all = all.cpu().numpy().astype(np.uint8)
    print('saving tensor to %s' % save_path)
    imageio.imwrite(save_path, all)


def untransformTensor(vggImageTensor):
    vggImageTensor = vggImageTensor.cpu()
    vggImageTensor = (vggImageTensor + 1) / 2
    vggImageTensor.clamp_(0, 1)

    vggImageTensor[vggImageTensor > 1.] = 1.
    vggImageTensor[vggImageTensor < 0.] = 0.
    vggImageTensor = vggImageTensor * 255
    return vggImageTensor


iterations = trainer.resume_test(checkpoint_directory, hyperparameters=config)
image_directory_test = os.path.join(image_directory, 'sample')
image_directory_test_real = os.path.join(image_directory, 'sample_real')
image_directory_test_pose = os.path.join(image_directory, 'sample_pose')
image_directory_test_nlp = os.path.join(image_directory, 'sample_nlp')
if not (os.path.exists(image_directory_test)):
    os.mkdir(image_directory_test)
if not (os.path.exists(image_directory_test_real)):
    os.mkdir(image_directory_test_real)

if not (os.path.exists(image_directory_test_pose)):
    os.mkdir(image_directory_test_pose)
if not (os.path.exists(image_directory_test_nlp)):
    os.mkdir(image_directory_test_nlp)


for it, (data_a, data_b) in enumerate(zip(test_loader_a, test_loader_b)):
    print(it)
    trainer.update_learning_rate()
    data_a = prepare_data(data_a)
    data_b = prepare_data(data_b)

    images_a, images_b = (data_a[0].cuda().detach()), (data_b[0].cuda().detach())
    pose_a, pose_b = (data_a[1].cuda().detach()), (data_b[1].cuda().detach())
    caps_a, caps_b = (data_a[2].cuda().detach().squeeze(dim=2)), (data_b[2].cuda().detach().squeeze(dim=2))
    caps_len_a, caps_len_b = (data_a[3].cuda().detach()), (data_b[3].cuda().detach())

    im_path_a = data_a[4][0]
    im_path_b = data_b[4][0]
    im_path_a_split = im_path_a.split('/')
    im_path_b_split = im_path_b.split('/')
    im_path_a = im_path_a_split[-2] + '_' + im_path_a_split[-1]
    im_path_b = im_path_b_split[-2] + '_' + im_path_b_split[-1]

    im_path_a_final = im_path_a[:-4]
    im_path_b_final = im_path_b[:-4]

    with torch.no_grad():
        test_image_outputs = trainer.test_sample_modify_adj(images_a, images_b, pose_a, pose_b, caps_a, caps_len_a,
                                                            caps_b, caps_len_b, ixtoword_test, wordtoix_test)

    img_all_a_real = untransformTensor(images_a.cpu())

    result_a_pose = test_image_outputs[1]
    img_all_a_pose = untransformTensor(result_a_pose.cpu())

    result_a_nlp = test_image_outputs[2]
    img_all_a_nlp = untransformTensor(result_a_nlp.cpu())

    writeTensor('%s/%s.jpg' % (image_directory_test_real, im_path_a_final), img_all_a_real, nRow=images_a.shape[0])
    writeTensor('%s/%s.jpg' % (image_directory_test_pose, im_path_a_final), img_all_a_pose, nRow=images_a.shape[0])
    writeTensor('%s/%s.jpg' % (image_directory_test_nlp, im_path_a_final), img_all_a_nlp, nRow=images_a.shape[0])

    result_a = test_image_outputs[5:10]

    for kk in range(len(result_a)):
        result_a_this = result_a[kk]
        img_all_a = untransformTensor(result_a_this.cpu())
        writeTensor('%s/%s_%d.jpg' % (image_directory_test, im_path_a_final, kk), img_all_a, nRow=images_a.shape[0])

