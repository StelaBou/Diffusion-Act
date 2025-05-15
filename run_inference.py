
import argparse
import sys
import torch # type: ignore
from torchvision import transforms # type: ignore
import os
import numpy as np
from PIL import Image
sys.path.append(".")
sys.path.append("..")
current_dir = '.'
sys.path.append(os.path.join(current_dir, 'libs', 'reenactment', 'EMOCA'))

from libs.templates import *
from libs.utils import make_path, save_image, torch_range_1_to_255, get_gaze_position, draw_stickman
from libs.ffhq_cropping import align_crop_image
from libs.reenactment.EMOCA.gdl_apps.EMOCA.calculate_emoca import extract_emoca_params
from libs.reenactment.EMOCA.gdl_apps.EMOCA.utils.load import load_model
from libs.reenactment.EMOCA.gdl_apps.EMOCA.utils.io import get_shape
from libs.reenactment.gaze_estimation.model import gaze_network
from libs.reenactment.controlnet import ControlNet, GlobalNoise
from libs.reenactment.face_models.landmarks_estimation import LandmarksEstimation


class Inference():

	def __init__(self, args) -> None:

		if os.path.exists(args.model_path):
			self.align_images = True
			self.method = "controlnet_no_shift"
			self.shape_model_type = "Emoca"
			self.add_gaze = True
			self.train_diffusion = True
			self.T_render = 20
			self.T_xt = 50
			self.model_path = args.model_path
		else:
			print('The provided moddel {} does not exist'.format(args.model_path))
			exit()

		self.load_models()

	def load_models(self):

		# Load EMOCA
		path_to_models = './pretrained_models/EMOCA/models'
		model_name = 'EMOCA_v2_lr_mse_20'
		mode = 'detail'
		self.emoca, conf = load_model(path_to_models, model_name, mode)
		self.emoca.cuda()
		self.emoca.eval()

		# Load gaze network
		self.gaze_model = gaze_network().cuda()
		pre_trained_model_path = './pretrained_models/gaze_estimation_model.tar'
		ckpt = torch.load(pre_trained_model_path)
		self.gaze_model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
		self.gaze_model.eval()  # change it to the evaluation mode

		self.landmarks_est = LandmarksEstimation(type = '2D')

		# Initialize diffusion model
		conf = ffhq256_autoenc() # ffhq256_autoenc_latent ffhq256_autoenc
		print(' -- Load DiffAE encoder model: {}'.format(conf.name))
		
		self.net = ControlNet(pretrained_model = f'checkpoints/{conf.name}/last.ckpt', method = self.method).cuda()
		ckpt = torch.load(self.model_path, map_location='cpu')
		print('best_val_loss ', ckpt['best_val_loss'])
		if 'global_noise' in ckpt.keys():
			self.global_noise = GlobalNoise().cuda()
			self.global_noise.load_state_dict(ckpt['global_noise'], strict=False)
			self.global_noise.eval()
		else:
			self.global_noise = None

		
		print('Load new diffusion model')
		self.diff_model = LitModel(conf)  # experiment.py
		self.diff_model.load_state_dict(ckpt['state_dict_diff'], strict=False)
		self.diff_model.ema_model.eval()
		self.diff_model = self.diff_model.cuda()
		self.diff_model.model.eval()
		self.net.load_state_dict(ckpt['state_dict_net'], strict=True)

	def align_image(self, image, landmarks):
		
		image_aligned = torch.zeros(image.shape).cuda()
		for i in range(image.shape[0]):
			image_np = (image[i] + 1) / 2 * 255.0
			image_np = image_np.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
			img = align_crop_image(image_np, landmarks)
			image_tensor = torch.tensor(np.transpose(img, (2,0,1))).float().div(255.0)	
			max_val = 1; min_val = -1
			image_tensor = image_tensor * (max_val - min_val) + min_val
			image_aligned[i] = image_tensor
	
		return image_aligned

	def load_aligned_image(self, image_path):
		self.transforms_dict = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		image = Image.open(image_path)
		image = image.convert('RGB')
		image = self.transforms_dict(image).unsqueeze(0).cuda()
		
		landmarks = self.get_landmarks(image)
		landmarks = landmarks[0].detach().cpu().numpy()
		image = self.align_image(image, landmarks)
		landmarks = self.get_landmarks(image) # get landmarks from aligned image
		
		return image, landmarks	

	def get_landmarks(self, frame):
		frame_255 = torch_range_1_to_255(frame)
		with torch.no_grad():
			landmarks = self.landmarks_est.get_landmarks_from_batch(frame_255) # torch tensor batch x 68 x 2
		return landmarks

	def get_landmarks_emoca(self, params):
		coefficients_gt = {}
		coefficients_gt['posecode'] = params['posecode']
		coefficients_gt['expcode'] = params['expcode']
		coefficients_gt['shapecode'] = params['shapecode']
		coefficients_gt['cam'] = params['cam']
		landmarks2d_gt, _, _ = get_shape(self.emoca, coefficients_gt)
		return landmarks2d_gt

	def get_stickman(self, image, landmarks):
	
		with torch.no_grad():
			
			params, _, _ = extract_emoca_params(self.emoca, image, landmarks.detach().cpu().numpy())
			landmarks_3dmm = self.get_landmarks_emoca(params)
			
			target_right_eye, target_left_eye = get_gaze_position(image, landmarks, landmarks_3dmm, self.gaze_model)
			
			stickman = draw_stickman(landmarks_3dmm, right_eye = target_right_eye, left_eye = target_left_eye).cuda().float()
		
			return stickman
	
	def forward_image(self, source_img, target_img, source_landmarks, target_landmarks, xT):
		with torch.no_grad():
			
			params_target, _, _ = extract_emoca_params(self.emoca, target_img, target_landmarks.detach().cpu().numpy())
			target_landmarks_3dmm = self.get_landmarks_emoca(params_target)
					
			params_source, _, _ = extract_emoca_params(self.emoca, source_img, source_landmarks.detach().cpu().numpy())
			coefficients_gt = {}
			coefficients_gt['posecode'] = params_target['posecode']
			coefficients_gt['expcode'] = params_target['expcode']
			coefficients_gt['shapecode'] = params_source['shapecode']
			coefficients_gt['cam'] = params_target['cam']
			
			target_landmarks_3dmm = self.get_landmarks_emoca(coefficients_gt)

			target_right_eye, target_left_eye = get_gaze_position(target_img, target_landmarks, target_landmarks_3dmm, self.gaze_model) 
			
			stickman_target = draw_stickman(target_landmarks_3dmm, right_eye = target_right_eye, left_eye = target_left_eye).cuda().float()			
			z_edit = self.net(source_img, stickman_target, None) 
			reenacted_img = self.diff_model.render(xT, z_edit, T=self.T_render, train_diff = True) #model = self.diff_model.model) # use this when trained diffusion Check base.py --> ddim_sample_loop_progressive
			
		return reenacted_img


	def run_inference_image_pair(self, source_path, target_path, output_path):

		make_path(output_path)

		self.net.eval(); self.diff_model.eval()
		source_img, source_landmarks = self.load_aligned_image(source_path)
		target_img, target_landmarks = self.load_aligned_image(target_path)

		stickman_source = self.get_stickman(source_img, source_landmarks) 
		z_source_hat = self.net(source_img, stickman_source, None) 
		xT = self.diff_model.encode_stochastic(source_img, z_source_hat, T=self.T_xt, model = self.diff_model.ema_model) 

		reenacted_img = self.forward_image(source_img, target_img, source_landmarks, target_landmarks, xT)

		out_filename = "{}_{}.png".format(os.path.splitext(os.path.basename(source_path))[0],  os.path.splitext(os.path.basename(target_path))[0])
		save_image(reenacted_img, os.path.join(output_path, out_filename))
	

def main():
	"""
	Inference script. Generate reenactment resuls.
	Input: 
		--source image: 			Reenact the source image 
		--target images:  			Driving image 
	Output:
		--reenacted image			

	Options:
		######### General ###########
		--source_path						: Path to source frame. Type: image (.png or .jpg)
		--target_path						: Path to target frame .Type: image (.png or .jpg)
		--output_path						: Path to save the generated images
	
	Example:

	python run_inference.py --source_path ./demo_run/source.png \
							--target_path ./demo_run/target.png \
							--output_path ./demo_run/results 

	"""
	parser = argparse.ArgumentParser(description="inference script")

	######### General ###########
	parser.add_argument('--source_path', type=str, required = True, help="path to source identity")
	parser.add_argument('--target_path', type=str, required = True, help="path to target pose")
	parser.add_argument('--output_path', type=str, required = True, help="path to save the results")

	parser.add_argument('--model_path', type=str, default = './pretrained_models/diffusionact_model.pt', help="path to pretrained model")

	

	
	# Parse given arguments
	args = parser.parse_args()	

	inference = Inference(args)
	inference.run_inference_image_pair(args.source_path, args.target_path, args.output_path)

	print("************************ Done **************************")
	


if __name__ == '__main__':
	main()
