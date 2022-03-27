from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
try:
    from itertools import izip as zip
except ImportError:
    pass


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

def util_cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

class Ac_ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(Ac_ResnetBlock, self).__init__()
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s



## the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, nlabels, size, params, embed_size=256, nfilter=64, nfilter_max=1024, size2=8):
        super(Discriminator, self).__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size2
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.conv_img = nn.Conv2d(input_dim, 1 * nf, 3, padding=1)
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [Ac_ResnetBlock(nf, nf*2)]
        blocks += [Ac_ResnetBlock(nf*2, nf * 2)]
        nf = nf * 2

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [nn.AvgPool2d(3, stride=2, padding=1),
                       Ac_ResnetBlock(nf0, nf1),
                       Ac_ResnetBlock(nf1, nf1)]

        self.resnet = nn.Sequential(*blocks)
        self.final_predict = nn.Conv2d(nf1, nlabels, 1, 1, 0)
        self.gan_type = params['gan_type']

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        out = self.conv_img(x)
        out_feature = self.resnet(out)
        out_real_fake = self.final_predict(out_feature)
        out_real_fake = out_real_fake[:, y.long(), :, :]
        return out_real_fake, out_feature

    def gradient_penalty(self, x, y, class_info):
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = util_cuda(torch.rand(shape))
        z = x + alpha * (y - x)

        z = util_cuda(Variable(z, requires_grad=True))
        o = self.forward(z, class_info)[0]
        g = grad(o, z, grad_outputs=util_cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
        return gp

    def calc_dis_loss(self, input_fake, input_fake_class, input_real, input_real_class):
        outs0, _ = self.forward(input_fake, input_fake_class)
        outs1, _ = self.forward(input_real, input_real_class)
        loss = 0

        if self.gan_type == 'lsgan':
            loss += torch.mean((outs0 - 0) ** 2) + torch.mean((outs1 - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(outs0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(outs1.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(outs0), all0) +
                               F.binary_cross_entropy(F.sigmoid(outs1), all1))
        elif self.gan_type == 'wgan':
            wd = torch.mean(outs1) - torch.mean(outs0)
            gp = self.gradient_penalty(input_real.data, input_fake.data, input_real_class)
            loss += -wd + gp * 10.0
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, input_fake_class):
        outs0, _ = self.forward(input_fake, input_fake_class)
        loss = 0
        if self.gan_type == 'lsgan':
            loss += torch.mean((outs0 - 1) ** 2)  # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(outs0.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(outs0), all1))
        elif self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


class Image_Encoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, norm, activ, pad_type):
        super(Image_Encoder, self).__init__()
        origin_dim = dim
        self.model_stage1 = []
        self.model_stage1 += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_stage1 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim = dim * 2
        self.model_stage1 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim = dim * 2
        self.model_stage1 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim = dim * 2

        self.model_stage2 = []
        self.model_stage2 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim = dim * 2
        self.model_stage2 += [nn.AdaptiveAvgPool2d(1)]
        self.model_stage2 += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model_stage1 = nn.Sequential(*self.model_stage1)
        self.model_stage2 = nn.Sequential(*self.model_stage2)
        self.output_dim = dim

        self.emb_features = conv1x1(8*origin_dim, style_dim)

    def forward(self, x):
        feature = self.model_stage1(x)
        feature_final = self.model_stage2(feature)
        feature_final = feature_final.view(feature_final.shape[0], -1)

        global_feature = self.emb_features(feature)
        return global_feature, feature_final


class Pose_Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(Pose_Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='adain', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero',
                 use_bias=True, spectral1=False, spectral2=False,
                 activation_first=False, norm_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        self.norm_first = norm_first
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        self.spectral1 = spectral1
        self.spectral2 = spectral2
        norm_dim = output_dim
        self.norm_name = norm
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if self.spectral1 or self.spectral2:
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm and not(self.spectral2):
                x = self.norm(x)
        elif self.norm_first:
            if self.norm and not(self.spectral2):
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
        else:
            x = self.conv(self.pad(x))
            if self.norm and not(self.spectral2):
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3, relu5_3]


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Pose_encoder_and_decoder(nn.Module):
    def __init__(self, input_dim, style_dim, params, n_res=2):
        super(Pose_encoder_and_decoder, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        ## the spatial encoder
        self.enc_content = Pose_Encoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        ## the decoder
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        num_adain_para = self.get_num_adain_params(self.dec)
        print('adain paramter number: ', num_adain_para)

        ## the mlp to compute AdaIN parameter
        self.mlp = MLP(style_dim, num_adain_para, mlp_dim, 3, norm='none', activ=activ)

        ## the attribute-vector generator
        self.mlp2 = MLP(style_dim*2, style_dim, style_dim, 2, norm='none', activ=activ)

    def forward(self, images, class_feature):
        class_feature = class_feature.squeeze(-1).squeeze(-1)

        content = self.enc_content.forward(images)
        adain_params = self.mlp.forward(class_feature)
        self.assign_adain_params(adain_params, self.dec)
        output_images = self.dec.forward(content)
        return output_images

    def forward_residual(self, image_feature, text_feature):
        final_feature = torch.cat([image_feature, text_feature], dim=1)
        changed_feature_residual = self.mlp2.forward(final_feature)
        changed_feature = image_feature + changed_feature_residual
        return changed_feature

    def forward_residual_test(self, image_feature, text_feature, alpha):
        final_feature = torch.cat([image_feature, text_feature], dim=1)
        changed_feature_residual = self.mlp2.forward(final_feature)
        changed_feature = image_feature + changed_feature_residual * alpha
        return changed_feature, changed_feature_residual

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

