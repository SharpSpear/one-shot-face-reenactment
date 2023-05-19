import matplotlib

# matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from torch.nn import functional as F


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'HibridEncoder':
            encoder = psp_encoders.HibridEncoder(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            # encoder_ckpt = torch.load(model_paths['ir_se50'])
            encoder_ckpt = torch.load(model_paths['e4e'])
            encoder_ckpt = get_keys(encoder_ckpt, 'encoder')
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def conv_warper(self, layer, input, style, noise=None, isToRGB=False):
        # the conv should change
        conv = layer.conv
        batch, in_channel, height, width = input.shape

        style = style.view(batch, 1, in_channel, 1, 1)
        weight = conv.scale * conv.weight * style

        if conv.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )

        if conv.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, conv.out_channel, height, width)
            out = conv.blur(out)

        elif conv.downsample:
            input = conv.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, conv.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, conv.out_channel, height, width)

        if not isToRGB:
            out = layer.noise(out, noise=noise)
            out = layer.activate(out)

        return out

    def WPtoS(self, G, latent_WP):
        # 将W+仿射变换到S
        WP2S = []
        WP2S.append(G.conv1.conv.modulation(latent_WP[:, 0]))
        WP2S.append(G.to_rgb1.conv.modulation(latent_WP[:, 1]))
        i = 1
        for conv1, conv2, to_rgb in zip(G.convs[::2], G.convs[1::2], G.to_rgbs):
            WP2S.append(conv1.conv.modulation(latent_WP[:, i]))
            WP2S.append(conv2.conv.modulation(latent_WP[:, i + 1]))
            WP2S.append(to_rgb.conv.modulation(latent_WP[:, i + 2]))
            i += 2
        return WP2S

    def generateWithS(self, G, style_space, latent, noise):
        out = G.input(latent)
        out = self.conv_warper(G.conv1, out, style_space[0], noise[0])
        skip = self.conv_warper(G.to_rgb1, out, style_space[1], isToRGB=True)
        # skip = G.to_rgb1(out, style_space[1])

        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
        ):
            out = self.conv_warper(conv1, out, style_space[i], noise=noise1)
            out = self.conv_warper(conv2, out, style_space[i + 1], noise=noise2)
            temp = self.conv_warper(to_rgb, out, style_space[i + 2], isToRGB=True)
            skip = temp + to_rgb.upsample(skip)
            # temp = to_rgb(out, style_space[i + 2], skip)

            i += 3

        image = skip

        return image

    def forward(self, x, resize=True, latent_mask=None, input_code=False, only_input_w=False, randomize_noise=True,
                inject_latent=None, return_latents=False, only_return_latents=False, alpha=None):
        if input_code:
            w, s = x
        else:
            w, s = self.encoder(x)

        if only_return_latents:
            return [w, s]
            # normalize with respect to the center of an average face
            # if self.opts.start_from_latent_avg:
            #     if codes.ndim == 2:
            #         codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            #     else:
            #         codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        if randomize_noise:
            noise = [None] * self.decoder.num_layers
        else:
            noise = [
                getattr(self.decoder.noises, "noise_{}".format(i)) for i in range(self.decoder.num_layers)
            ]
        # 对w进行仿射变换到S
        w2s = self.WPtoS(self.decoder, w)
        # 将生成的S加到W变换后的S，得到最终用于生成latent
        if not only_input_w:
            for i in range(s.shape[1]):
                w2s[i] += s[:, i, :]
        # 通过latent生成图像
        images = self.generateWithS(self.decoder, w2s, w, noise)
        # 返回的latent由生成的w
        result_latent = [w, s]
        # if latent_mask is not None:
        #     for i in latent_mask:
        #         if inject_latent is not None:
        #             if alpha is not None:
        #                 codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
        #             else:
        #                 codes[:, i] = inject_latent[:, i]
        #         else:
        #             codes[:, i] = 0

        # input_is_latent = not input_code
        # images, result_latent = self.decoder([w],
        #                                      input_is_latent=input_is_latent,
        #                                      randomize_noise=randomize_noise,
        #                                      return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
