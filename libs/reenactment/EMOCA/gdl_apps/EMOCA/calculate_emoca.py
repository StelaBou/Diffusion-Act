import torch
import numpy as np

from skimage.transform import estimate_transform
import kornia
import math

from gdl.datasets.ImageDatasetHelpers import bbox2point
from gdl_apps.EMOCA.utils.io import run_emoca
from gdl_apps.EMOCA.rotation_converter import *
from libs.utils import torch_range_1_to_255


def get_image(image, kpt):
	scaling_factor = 1.0
	crop_size = 224
	scale = 1.25
	iscrop = True
	resolution_inp = 224

	image = torch_range_1_to_255(image)

	left = np.min(kpt[:, 0])
	right = np.max(kpt[:, 0])
	top = np.min(kpt[:, 1])
	bottom = np.max(kpt[:, 1])
	old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
	h = 256; w = 256
	if math.isnan(old_size):
		# Run original image
		left = 0
		right = h - 1
		top = 0
		bottom = w - 1
		old_size, center = bbox2point(left, right, top, bottom, type='kpt68')


	if isinstance(old_size, list):
		size = []
		src_pts = []
		for i in range(len(old_size)):
			size += [int(old_size[i] * scale)]
			src_pts += [np.array(
				[[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
				[center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
	else:
		
		size = int(old_size * scale)
		src_pts = np.array(
			[[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
			[center[0] + size / 2, center[1] - size / 2]])

	DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
	tform = estimate_transform('similarity', src_pts, DST_PTS)
	theta =  torch.tensor(tform.params, dtype=torch.float32).unsqueeze(0).cuda()

	image_tensor = image.clone()
	image_tensor = image_tensor.unsqueeze(0).cuda()
	image_tensor = image_tensor.div(255.)
	dst_image = kornia.geometry.warp_affine(image_tensor, theta[:,:2,:], dsize=(224, 224)) # Add geometry for kornia version '0.7.0'
	
	return dst_image.cuda()

'Batch torch tensor'
def extract_emoca_params(emoca, images, landmarks):
	
	p_tensor = torch.zeros(images.shape[0], 6).cuda()
	alpha_shp_tensor = torch.zeros(images.shape[0], 100).cuda()
	alpha_exp_tensor = torch.zeros(images.shape[0], 50).cuda()
	cam = torch.zeros(images.shape[0], 3).cuda()
	angles = torch.zeros(images.shape[0], 3).cuda()
	image_prepro_batch = torch.zeros(images.shape[0], 3, 224, 224).cuda()
	for batch in range(images.shape[0]):            
		image_prepro = get_image(images[batch].clone(), landmarks[batch])
		batch_emoca = {
			'image':		image_prepro
		}
		image_prepro_batch[batch] = image_prepro[0]
		codedict = run_emoca(emoca, batch_emoca)
		p_tensor[batch] = codedict['posecode'][0]
		alpha_shp_tensor[batch] = codedict['shapecode'][0]
		alpha_exp_tensor[batch] = codedict['expcode'][0]
		cam[batch] = codedict['cam'][0]
		pose = codedict['posecode'][:,:3]
		pose = rad2deg(batch_axis2euler(pose))
		angles[batch] = pose
	out_dict = {}
	out_dict['posecode'] = p_tensor
	out_dict['expcode'] = alpha_exp_tensor
	out_dict['shapecode'] = alpha_shp_tensor
	out_dict['cam'] = cam
	batch_emoca['image'] = image_prepro_batch
	return out_dict, batch_emoca, angles

def run_emoca_mine(emoca, images):
	codedict = run_emoca(emoca, images)
	pose = codedict['posecode'][:,:3]
	pose = rad2deg(batch_axis2euler(pose))

	return codedict, pose



