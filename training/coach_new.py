import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from configs.paths_config import model_paths
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from configs import data_configs
from datasets.images_dataset_celeb_HDTF import ImagesDataset
from criteria.lpips.lpips import LPIPS
from new_criteria import id_loss, id_latent_loss, regularization_loss, gradient_variance_loss
from models.psp import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger

from new_criteria.faceParsing.model import BiSeNet



random.seed(0)
torch.manual_seed(0)


class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device
        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        # Initialize loss
        if self.opts.l2_lambda > 0:
            self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
        if self.opts.gradient_variance_lambda > 0:
            self.gradient_variance_loss = gradient_variance_loss.GradientVariance(patch_size=8)
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.id_latent_lambda > 0:
            self.id_latent_loss = id_latent_loss.IDLatentLoss()
        if self.opts.regularization_lambda > 0:
            self.reg_loss = regularization_loss.RegLoss(self.opts.s_lambda)
        if self.opts.feature_reconstruction_lambda > 0:
            # 加载人脸分割模型
            self.biSeNet = BiSeNet(n_classes=19)
            self.biSeNet.cuda()
            self.biSeNet.load_state_dict(torch.load(model_paths['face_parsing']))
            self.biSeNet.eval()
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)


        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch[0])

                x_refer, y1_refer, y2_refer, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1 = self.forward(batch)

                loss, encoder_loss_dict = self.calc_loss(
                    x_refer, y1_refer, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1)

                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images_all(x_refer, y1_refer, y2_refer, x_x, x_id, x_y1, x_y2,title='images/train/')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:  # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x_refer, y1_refer, y2_refer, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1 = self.forward(batch)
                loss, cur_encoder_loss_dict = self.calc_loss(x_refer, y1_refer, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            self.parse_and_log_images_all(x_refer, y1_refer, y2_refer, x_x, x_id, x_y1, x_y2,
                                          title='images/test/', subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            raise ValueError('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(main_root=dataset_args['train_source_root'],
                                      transform=transforms_dict['transform_gt_train'])
        test_dataset = ImagesDataset(main_root=dataset_args['test_source_root'],
                                     transform=transforms_dict['transform_test'])
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def get_mask(self, img):

        with torch.no_grad():
            # 从256上采样到512
            image = F.upsample(img, scale_factor=2, mode='bilinear')
            out = self.biSeNet(image)[0]
            # 将mask从512下采样到256
            out = F.max_pool2d(out, 2)
            parsing = torch.argmax(out, dim=1).type(torch.uint8)

        mask = (parsing == 2) | (parsing == 3)  # 眉毛
        mask += (parsing == 4) | (parsing == 5)  # 眼睛
        mask += (parsing == 11) | (parsing == 12) | (parsing == 13)  # 嘴巴

        # 膨胀mask
        # ksizze 必须为奇数
        ksize = 21
        # 首先为原图加入 padding，防止图像尺寸缩小
        B, H, W = mask.shape
        C = 1
        mask = mask.reshape(B, C, H, W)
        pad = (ksize - 1) // 2
        bin_img = F.pad(mask, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最大的值，i.e., 0
        mask_dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

        return mask_dilated

    def reconstruction_loss(self, x, y):
        # x和y为图像，计算x与y的重构损失
        loss = 0
        if self.opts.l2_lambda > 0:
            loss_l2 = self.mse_loss(x, y)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(x, y)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.gradient_variance_lambda > 0:
            loss_gradient = self.gradient_variance_loss(x, y)
            loss += loss_gradient * self.opts.gradient_variance_lambda
        return loss

    def feature_recconstruction_loss(self, x, y1, x_x, x_y1):
        mask_x = self.get_mask(x)
        mask_y1 = self.get_mask(y1)
        loss = self.reconstruction_loss(x * mask_x, x_x * mask_x) + \
               self.reconstruction_loss(y1 * mask_y1, x_y1 * mask_y1)
        return loss

    def calc_loss(self, x, y1, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1):
        loss_dict = {}
        loss = 0

        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))

            for i in dims_to_discriminate:
                w = w_x[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc * self.opts.w_discriminator_lambda)
            loss += self.opts.w_discriminator_lambda * loss_disc


        loss_self = self.reconstruction_loss(x, x_x)
        loss_dict['loss_self'] = float(loss_self)
        loss += loss_self

        loss_reenact = self.reconstruction_loss(y1, x_y1)
        loss_dict['loss_reenact'] = float(loss_reenact)
        loss += loss_reenact

        if self.opts.id_lambda > 0:
            loss_id = self.id_loss(x_id, x, x_y2)
            loss_dict['loss_id'] = float(loss_id * self.opts.id_lambda)
            loss += loss_id * self.opts.id_lambda
        if self.opts.id_latent_lambda > 0:
            loss_id_latent = self.id_latent_loss(w_x, w_x_y2)
            loss_dict['loss_latent_id'] = float(loss_id_latent * self.opts.id_latent_lambda)
            loss += loss_id_latent * self.opts.id_latent_lambda
        if self.opts.feature_reconstruction_lambda > 0:
            loss_feature_recon = self.feature_recconstruction_loss(x, y1, x_x, x_y1)
            loss_dict['loss_feature_recon'] = float(loss_feature_recon * self.opts.feature_reconstruction_lambda)
            loss += loss_feature_recon * self.opts.feature_reconstruction_lambda
        if self.opts.regularization_lambda > 0:
            if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 18:  # delta regularization loss
                stage = self.net.encoder.progressive_stage.value
                loss_reg = self.reg_loss(w_x, s_y1, stage)
                loss_dict['loss_reg'] = float(loss_reg * self.opts.regularization_lambda)
                loss += loss_reg * self.opts.regularization_lambda

        loss_dict['loss'] = float(loss)

        return loss, loss_dict

    def forward(self, batch):
        # 对应论文中的变量 x=S,y1=D1,y2=D2
        x_refer = batch[0].cuda()
        y1_refer = batch[1].cuda()
        y2_refer = batch[2].cuda()
        # 通过Encoder得到所有的w和s
        w_x, s_x = self.net.forward(x_refer, only_return_latents=True)
        w_y1, s_y1 = self.net.forward(y1_refer, only_return_latents=True)
        w_y2, s_y2 = self.net.forward(y2_refer, only_return_latents=True)
        # 组合不同的w和s通过Decoder生成图像（无梯度）

        x_x = self.net.forward([w_x, s_x], input_code=True)
        x_id = self.net.forward([w_x, None], input_code=True, only_input_w=True)
        x_y1 = self.net.forward([w_x, s_y1], input_code=True)
        x_y2 = self.net.forward([w_x, s_y2], input_code=True)
        # 对y2驱动后的图像再次通过Encoder得到相应的w
        w_x_y2, _ = self.net.forward(x_y2, only_return_latents=True)

        return x_refer, y1_refer, y2_refer, x_x, x_id, x_y1, x_y2, w_x, w_x_y2, s_y1

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def parse_and_log_images_all(self, x, y1, y2, x_x, x_id, x_y1, x_y2, title, subscript=None, display_count=3):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'source': common.tensor2im(x[i]),
                'source_rec': common.tensor2im(x_x[i]),
                'source_id': common.tensor2im(x_id[i]),
                'driven1': common.tensor2im(y1[i]),
                'source_driven1': common.tensor2im(x_y1[i]),
                'driven2': common.tensor2im(y2[i]),
                'source_driven2': common.tensor2im(x_y2[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_all(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):
        loss_dict = {}
        # x, _ = batch
        x = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _,_ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.net.decoder.get_latent(sample_z)
        fake_w = self.net.encoder(x)[0]
        # print(fake_w.size())
        # print(fake_w)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
