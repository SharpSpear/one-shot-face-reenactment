import torch
from torch import nn
from utils import common
from new_criteria import id_loss, id_latent_loss,feature_recon_loss,regularization_loss,gradient_variance_loss
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp

class Coach:
	def __init__(self, opts):
		self.opts = opts

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss()
		if self.opts.feature_reconstruction_lambda > 0:
			self.feature_recon_loss_x = feature_recon_loss.FeatureReconLoss()
			self.feature_recon_loss_y = feature_recon_loss.FeatureReconLoss()
		if self.opts.regularization_lambda > 0:
			self.recon_loss = regularization_loss.RegLoss()
			# self.feature_recon_loss_y = feature_recon_loss.FeatureReconLoss()
		if self.opts.gradient_variance_lambda>0:
			self.gradient_variance_loss=gradient_variance_loss.GradientVariance()
	def calc_loss(self,x, x_hat, y, y_hat, latent_x,latent_y,S_D):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id = self.id_loss(y_hat, y, x_hat)
			loss_dict['loss_id'] = float(loss_id)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2_y = self.mse_loss(y_hat, y)
			loss_l2_x = self.mse_loss(x_hat, x)
			loss_l2=loss_l2_y+loss_l2_x
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips_y = self.lpips_loss(y_hat, y)
			loss_lpips_x = self.lpips_loss(x_hat, x)
			loss_lpips=loss_lpips_y+loss_lpips_x
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.gradient_variance_lambda>0:
			loss_gradient_y=self.gradient_variance_loss(y_hat, y)
			loss_gradient_x=self.gradient_variance_loss(x_hat, x)
			loss_gradient=loss_gradient_y+loss_gradient_x
			
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent_x, latent_y)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.feature_reconstruction_lambda > 0:
			loss_feature_recon_y = self.feature_recon_loss_y(y_hat, y)
			loss_feature_recon_x = self.feature_recon_loss_x(x_hat, x)
			loss_feature_recon=loss_feature_recon_y+loss_feature_recon_x
			loss_dict['loss_feature_recon'] = float(loss_feature_recon)
			loss += loss_feature_recon * self.opts.feature_reconstruction_lambda
		if self.opts.regularization_lambda > 0:
			loss_recon = self.recon_loss(latent_y,S_D)
			loss_dict['loss_recon'] = float(loss_recon)
			loss += loss_recon * self.opts.regularization_lambda

		return loss