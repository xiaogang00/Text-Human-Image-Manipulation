from networks import Discriminator, Pose_encoder_and_decoder, Image_Encoder
from utils_test import weights_init, get_model_list
from text_model import RNN_ENCODER_attention
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        ## the network contains the spatial encoder, the decoder, the mlp to compute AdaIN parameter and the attribute-vector generator
        self.gen_a = Pose_encoder_and_decoder(hyperparameters['input_dim_a'], hyperparameters['TEXT_EMBEDDING_DIM'], hyperparameters['gen'])

        ## the text encoder
        self.text_encoder = RNN_ENCODER_attention(hyperparameters['dataset_word_num'], hyperparameters, nhidden=hyperparameters['TEXT_EMBEDDING_DIM'])

        ## the image encoder
        self.image_encoder = Image_Encoder(3, 64, hyperparameters['TEXT_EMBEDDING_DIM'], norm='none', activ='relu', pad_type='reflect')

        self.instancenorm_list = [nn.InstanceNorm2d(64, affine=False),
                                  nn.InstanceNorm2d(128, affine=False),
                                  nn.InstanceNorm2d(256, affine=False),
                                  nn.InstanceNorm2d(512, affine=False),
                                  nn.InstanceNorm2d(512, affine=False)]

    def drawCaption(self, captions, ixtoword, h, w):
        off1 = 0
        off2 = 0
        caption_img = []
        for mm in range(captions.shape[0]):
            convas = np.zeros((h, w, 3))
            cap = captions[mm].data.cpu().numpy()
            img_txt = Image.fromarray(convas.astype(np.uint8))
            fnt = ImageFont.load_default().font

            vis_size = 50
            d = ImageDraw.Draw(img_txt)
            row_num = 1
            col_num = 0
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')

                if j % 2 == 0:
                    row_num += 1
                    col_num = 0
                d.text(((col_num + off1) * (vis_size + off2), 8 * (row_num - 1)), '%s' % (word),
                       font=fnt, fill=(255, 255, 255, 255))
                col_num += 1
            img_txt = np.array(img_txt)
            if len(img_txt.shape) == 2:
                img_txt = np.dstack((img_txt, img_txt, img_txt))

            img_txt[:, w-3:w-1, :] = 255
            img_txt[0:3, :, :] = 255
            img_txt[h-3:h-1, :, :] = 255

            img_txt = torch.Tensor(img_txt).unsqueeze(0)
            caption_img.append(img_txt)
        caption_img = torch.cat(caption_img, dim=0)
        caption_img = torch.Tensor(caption_img).permute(0, 3, 1, 2)
        caption_img = caption_img * 1.0 / 255
        caption_img = caption_img * 2 - 1
        return caption_img.cuda()

    def word_transform(self, cap, ixtoword, wordtoix, color_target):
        cap_transform = cap.clone()
        cap_this = cap.data.cpu().numpy()
        color_list = ['black', 'gray', 'blue', 'red', 'yellow', 'green', 'white', 'grey', 'pink',
                      'purple', 'brown', 'orange', 'coloured', 'dark', 'khaki']
        flag = 1
        word_list = []
        for j in range(len(cap_this)):
            if cap_this[j] == 0:
                break
            word = ixtoword[cap_this[j]].encode('ascii', 'ignore').decode('ascii')
            if (word in color_list) and (flag == 1):
                word = color_target
                flag = 0
            word_list.append(word)
            ix = wordtoix[word]
            cap_transform[j] = ix
        return cap_transform

    def test_sample_modify_adj(self, x_a, x_b, x_a_pose, x_b_pose, captions_a, cap_lens_a, captions_b, cap_lens_b, ixtoword, wordtoix):
        self.eval()
        x_a_pose_save, x_b_pose_save = [], []
        x_a_caption_save, x_b_caption_save = [], []
        x_a_save, x_b_save = [], []
        x_a_recon_image, x_b_recon_image = [], []
        x_a_recon_text, x_b_recon_text = [], []

        color_list = ['black', 'gray', 'blue', 'red', 'yellow']
        x_a_result_list = []
        x_b_result_list = []
        for mm in range(len(color_list)):
            x_a_result_list.append([])
            x_b_result_list.append([])

        x_a_result_list_caption = []
        x_b_result_list_caption = []
        for mm in range(len(color_list)):
            x_a_result_list_caption.append([])
            x_b_result_list_caption.append([])

        for i in range(len(x_a)):
            hidden = self.text_encoder.init_hidden(1)
            words_embs_source, sent_emb_source = self.text_encoder.forward(captions_a[i].unsqueeze(0), cap_lens_a[i].view(1, ), hidden)
            words_embs_target, sent_emb_target = self.text_encoder.forward(captions_b[i].unsqueeze(0), cap_lens_b[i].view(1, ), hidden)
            _, image_feature_source = self.image_encoder.forward(x_a[i].unsqueeze(0))
            _, image_feature_target = self.image_encoder.forward(x_b[i].unsqueeze(0))

            x_a_recon_smaple_image = self.gen_a.forward(x_a_pose[i].unsqueeze(0), image_feature_source)
            x_b_recon_smaple_image = self.gen_a.forward(x_b_pose[i].unsqueeze(0), image_feature_target)

            x_a_recon_smaple_text = self.gen_a.forward(x_a_pose[i].unsqueeze(0), sent_emb_source)
            x_b_recon_smaple_text = self.gen_a.forward(x_b_pose[i].unsqueeze(0), sent_emb_target)

            x_a_recon_image.append(x_a_recon_smaple_image)
            x_b_recon_image.append(x_b_recon_smaple_image)
            x_a_recon_text.append(x_a_recon_smaple_text)
            x_b_recon_text.append(x_b_recon_smaple_text)
            x_a_save.append(x_a[i].unsqueeze(0))
            x_b_save.append(x_b[i].unsqueeze(0))
            x_a_pose_save.append(x_a_pose[i].unsqueeze(0))
            x_b_pose_save.append(x_b_pose[i].unsqueeze(0))
            caption_a_img = self.drawCaption(captions_a[i].unsqueeze(0), ixtoword, x_a.shape[2], x_a.shape[3])
            caption_b_img = self.drawCaption(captions_b[i].unsqueeze(0), ixtoword, x_b.shape[2], x_b.shape[3])
            x_a_caption_save.append(caption_a_img)
            x_b_caption_save.append(caption_b_img)

            ################################################################
            for mm in range(len(color_list)):
                transform_cap_a = self.word_transform(captions_a[i], ixtoword, wordtoix, color_list[mm])
                transform_cap_b = self.word_transform(captions_b[i], ixtoword, wordtoix, color_list[mm])
                _, sent_emb_source = self.text_encoder.forward(transform_cap_a.unsqueeze(0), cap_lens_a[i].view(1, ), hidden)
                _, sent_emb_target = self.text_encoder.forward(transform_cap_b.unsqueeze(0), cap_lens_b[i].view(1, ), hidden)

                sent_emb_source, _ = self.gen_a.forward_residual_test(image_feature_source, sent_emb_source, 1.0)
                sent_emb_target, _ = self.gen_a.forward_residual_test(image_feature_target, sent_emb_target, 1.0)

                x_a_recon_smaple_text2 = self.gen_a.forward(x_a_pose[i].unsqueeze(0), sent_emb_source)
                x_b_recon_smaple_text2 = self.gen_a.forward(x_b_pose[i].unsqueeze(0), sent_emb_target)

                x_a_result_list[mm].append(x_a_recon_smaple_text2)
                x_b_result_list[mm].append(x_b_recon_smaple_text2)

                caption_a_img = self.drawCaption(transform_cap_a.unsqueeze(0), ixtoword, x_a.shape[2], x_a.shape[3])
                caption_b_img = self.drawCaption(transform_cap_b.unsqueeze(0), ixtoword, x_b.shape[2], x_b.shape[3])
                x_a_result_list_caption[mm].append(caption_a_img)
                x_b_result_list_caption[mm].append(caption_b_img)

        x_a_save, x_b_save = torch.cat(x_a_save), torch.cat(x_b_save)
        x_a_pose_save, x_b_pose_save = torch.cat(x_a_pose_save), torch.cat(x_b_pose_save)
        x_a_caption_save, x_b_caption_save = torch.cat(x_a_caption_save), torch.cat(x_b_caption_save)
        x_a_recon_smaple_image, x_b_recon_smaple_image = torch.cat(x_a_recon_image), torch.cat(x_b_recon_image)
        x_a_recon_smaple_text, x_b_recon_smaple_text = torch.cat(x_a_recon_text), torch.cat(x_b_recon_text)

        for mm in range(len(color_list)):
            x_a_result_list[mm] = torch.cat(x_a_result_list[mm])
            x_b_result_list[mm] = torch.cat(x_b_result_list[mm])

        for mm in range(len(color_list)):
            x_a_result_list_caption[mm] = torch.cat(x_a_result_list_caption[mm])
            x_b_result_list_caption[mm] = torch.cat(x_b_result_list_caption[mm])
        self.train()
        return x_a_save, x_a_pose_save, x_a_caption_save, x_a_recon_smaple_image, x_a_recon_smaple_text, \
               x_a_result_list[0], x_a_result_list[1], x_a_result_list[2], x_a_result_list[3], x_a_result_list[4], \
               x_a_result_list_caption[0], x_a_result_list_caption[1],  \
               x_a_result_list_caption[2], x_a_result_list_caption[3], x_a_result_list_caption[4],\
               x_b_save, x_b_pose_save, x_b_caption_save, x_b_recon_smaple_image, x_b_recon_smaple_text, \
               x_b_result_list[0], x_b_result_list[1], x_b_result_list[2], x_b_result_list[3], x_b_result_list[4], \
               x_b_result_list_caption[0], x_b_result_list_caption[1], \
               x_b_result_list_caption[2], x_b_result_list_caption[3], x_b_result_list_caption[4]


    def test_sample_user_input(self, x_a, x_b, x_a_pose, x_b_pose, captions_a, cap_lens_a, captions_b, cap_lens_b, ixtoword, wordtoix):
        self.eval()
        x_a_save, x_b_save = [], []
        x_a_pose_save, x_b_pose_save = [], []
        x_a_recon_image, x_b_recon_image = [], []

        x_a_result_list_caption = []
        x_b_result_list_caption = []

        for i in range(len(x_a)):
            hidden = self.text_encoder.init_hidden(1)
            words_embs_source, sent_emb_source = self.text_encoder.forward(captions_a[i].unsqueeze(0), cap_lens_a[i].view(1, ), hidden)
            words_embs_target, sent_emb_target = self.text_encoder.forward(captions_b[i].unsqueeze(0), cap_lens_b[i].view(1, ), hidden)
            _, image_feature_source = self.image_encoder.forward(x_a[i].unsqueeze(0))
            _, image_feature_target = self.image_encoder.forward(x_b[i].unsqueeze(0))

            x_a_save.append(x_a[i].unsqueeze(0))
            x_b_save.append(x_b[i].unsqueeze(0))
            x_a_pose_save.append(x_a_pose[i].unsqueeze(0))
            x_b_pose_save.append(x_b_pose[i].unsqueeze(0))

            sent_emb_source, _ = self.gen_a.forward_residual_test(image_feature_source, sent_emb_source, 1.0)
            sent_emb_target, _ = self.gen_a.forward_residual_test(image_feature_target, sent_emb_target, 1.0)

            x_a_recon_smaple_text2 = self.gen_a.forward(x_a_pose[i].unsqueeze(0), sent_emb_source)
            x_b_recon_smaple_text2 = self.gen_a.forward(x_b_pose[i].unsqueeze(0), sent_emb_target)

            x_a_recon_image.append(x_a_recon_smaple_text2)
            x_b_recon_image.append(x_b_recon_smaple_text2)

            caption_a_img = self.drawCaption(captions_a[i].unsqueeze(0), ixtoword, x_a.shape[2], x_a.shape[3])
            caption_b_img = self.drawCaption(captions_b[i].unsqueeze(0), ixtoword, x_b.shape[2], x_b.shape[3])
            x_a_result_list_caption.append(caption_a_img)
            x_b_result_list_caption.append(caption_b_img)

        x_a_save, x_b_save = torch.cat(x_a_save), torch.cat(x_b_save)
        x_a_pose_save, x_b_pose_save = torch.cat(x_a_pose_save), torch.cat(x_b_pose_save)
        x_a_recon_smaple_image, x_b_recon_smaple_image = torch.cat(x_a_recon_image), torch.cat(x_b_recon_image)
        x_a_result_list_caption, x_b_result_list_caption = torch.cat(x_a_result_list_caption), torch.cat(x_b_result_list_caption)

        self.train()
        return x_a_save, x_a_pose_save, x_a_recon_smaple_image, \
               x_b_save, x_b_pose_save, x_b_recon_smaple_image, \
               x_a_result_list_caption, x_b_result_list_caption

    def resume_test(self, checkpoint_dir, hyperparameters):
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "image")
        state_dict = torch.load(last_model_name)
        self.image_encoder.load_state_dict(state_dict['a'])

        last_model_name = get_model_list(checkpoint_dir, "text")
        state_dict = torch.load(last_model_name)
        self.text_encoder.load_state_dict(state_dict['a'])
        print('Getting from iteration %d' % iterations)
        return iterations
