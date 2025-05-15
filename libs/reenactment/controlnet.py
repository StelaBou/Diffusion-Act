import torch
from torch import nn
import string

from libs.model.unet import BeatGANsEncoderConfig


class GlobalNoise(nn.Module):
	def __init__(self):
		super(GlobalNoise, self).__init__()

		self.noise = torch.nn.Parameter(torch.randn(3, 256, 256))



def load_my_state_dict(controlnet, state_dict):
 
	own_state = controlnet.state_dict()
	prefix = "ema_model.encoder."
	
	for name, param in state_dict.items():
		if prefix in name:
			param = param.data
			name_new = name.replace(prefix, "") 
			own_state[name_new].copy_(param)

	return own_state

class ControlNet(nn.Module):
	def __init__(self, pretrained_model = '', method = 'controlnet'):
		super(ControlNet, self).__init__()

		self.method = method
		
		use_checkpoint = False; enc_grad_checkpoint = False
		self.controlnet = BeatGANsEncoderConfig(
            image_size=256,
            in_channels=3,
            model_channels=128,
            out_hid_channels=512,
            out_channels=512,
            num_res_blocks=2,
            attention_resolutions=(None
                                   or (16,)),
            dropout=0.1,
            channel_mult= (1, 1, 2, 2, 4, 4, 4) or (1, 1, 2, 2, 4, 4),
            use_time_condition=False,
            conv_resample=True,
            dims=2,
            use_checkpoint=use_checkpoint or enc_grad_checkpoint,
            num_heads=1,
            num_head_channels=-1,
            resblock_updown=True,
            use_new_attention_order=False,
            pool='adaptivenonzero',
        ).make_model()

		self.zero_conv_1 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
		self.zero_conv_1 = self.zero_module(self.zero_conv_1)


		state = torch.load(pretrained_model, map_location='cpu')
		pretrained_state = load_my_state_dict(self.controlnet, state['state_dict'])
		self.controlnet.load_state_dict(pretrained_state, strict=True)


	
	def zero_module(self, module):
		"""
		Zero out the parameters of a module and return it.
		"""
		for p in module.parameters():
			p.detach().zero_()
		return module
		

	
	def forward(self, image, cond, return_2d_feature = False):
		
		cond = self.zero_conv_1(cond)
		input_ = image + cond # 3 x 256 x 256
		
		if return_2d_feature:
			h, h_2d = self.controlnet(input_, return_2d_feature = return_2d_feature) # Previously h_2d was 512 x 4 x 4, now in order to predict X_T I take 512 x 8 x 8
			return h, h_2d
		else:
			h = self.controlnet(input_, return_2d_feature = return_2d_feature) # Previously h_2d was 512 x 4 x 4, now in order to predict X_T I take 512 x 8 x 8
			return h

