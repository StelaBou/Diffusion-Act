
from torchvision import utils as torch_utils
import os
import numpy as np
import torch
from torchvision import transforms
from skimage import transform
import cv2



def find_affine_transformation(poses, h_img=256, w_img=256, scale = False):
    """
    Function return matrix of affine transformation to use in torch.nn.functional.affine_grid

    Input:
    facial_keypoints: np.array of size (5, 2) - coordinates of key facial points in pixel coords
                      right eye, left eye, nose, right mouse, left mouse
    h_img: int, height of input image
    w_img: int, width of input image
    returns: np.array of size (2, 3) - affine matrix
    """
    # poses = (poses.detach() + 1) / 2

    right_eye = list(range(36 , 42))
    left_eye = list(range(42, 48))
    nose = [30]
    right_mouth = [48]
    left_mouth = [54]
    
    if torch.is_tensor(poses):
        keypoints = poses.cpu().numpy().reshape(-1, 68, 2)
    else:
        keypoints = poses.reshape(-1, 68, 2)

    facial_keypoints = np.concatenate([
        keypoints[:, right_eye].astype('float32').mean(1, keepdims=True), # right eye
        keypoints[:, left_eye].astype('float32').mean(1, keepdims=True), # left eye
        keypoints[:, nose].astype('float32'), # nose
        keypoints[:, right_mouth].astype('float32'), # right mouth
        keypoints[:, left_mouth].astype('float32'), # left mouth
    ], 1)
    
    if scale:
        facial_keypoints[:, 0, 0] -= 20 # right eye (left opws to vlepw sthn eikona )
        facial_keypoints[:, 1, 0] += 20 # left eye (right opws to vlepw sthn eikona )
        facial_keypoints[:, 3, 1] += 20
        facial_keypoints[:, 4, 1] += 20
    
    h_grid = 112
    w_grid = 112

    src = np.array([
        [35.343697, 51.6963] ,
        [76.453766, 51.5014],
        [56.029396, 71.7366],
        [39.14085 , 92.3655],
        [73.18488 , 92.2041]], dtype=np.float32)

    affine_matrices = []

    for facial_keypoints_i in facial_keypoints:
        tform = transform.estimate_transform('similarity', src, facial_keypoints_i)
        affine_matrix = tform.params[:2, :]

        affine_matrices.append(affine_matrix)

    affine_matrices = np.stack(affine_matrices, axis=0)
   
    # do transformation for grid in [-1, 1]
    affine_matrices[:, 0, 0] = affine_matrices[:, 0, 0]*w_grid/w_img
    affine_matrices[:, 0, 1] = affine_matrices[:, 0, 1]*h_grid/w_img
    affine_matrices[:, 0, 2] = (affine_matrices[:, 0, 2])*2/w_img + affine_matrices[:, 0, 1] + affine_matrices[:, 0, 0] - 1
    affine_matrices[:, 1, 0] = affine_matrices[:, 1, 0]*w_grid/h_img
    affine_matrices[:, 1, 1] = affine_matrices[:, 1, 1]*h_grid/h_img
    affine_matrices[:, 1, 2] = (affine_matrices[:, 1, 2])*2/h_img + affine_matrices[:, 1, 0] + affine_matrices[:, 1, 1] - 1
    
    affine_matrices = torch.from_numpy(affine_matrices).float().cuda()
    
    return affine_matrices

def estimate_gaze_direction(image, landmarks, model):
	# Get transformation
	trans = transforms.Compose([
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]),
	])

	b, c, h, w = image.shape

	affine_matrices = find_affine_transformation(landmarks, h, w, scale = False)
	batch_size = image.shape[0]
	grid = torch.nn.functional.affine_grid(affine_matrices, torch.Size((batch_size, 3, 224, 224)))      
	warped_image = torch.nn.functional.grid_sample(image, grid)
	warped_image_keep = warped_image
	warped_image = warped_image.clamp(-1,1).add(1.0).div(2.0) # [-1,1] to [0,1]
	input_var = trans(warped_image)
	
	pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
	
	return pred_gaze, warped_image_keep

def get_eye_width_height_batch(facial_landmarks_batch):
	# Left eye landmarks indices (assuming the facial landmarks follow the dlib or similar conventions)
	left_eye_indices = [36, 37, 38, 39, 40, 41]

	# Right eye landmarks indices
	right_eye_indices = [42, 43, 44, 45, 46, 47]

	# Extract left and right eye landmarks for the entire batch
	left_eye_landmarks_batch = facial_landmarks_batch[:, left_eye_indices, :]
	right_eye_landmarks_batch = facial_landmarks_batch[:, right_eye_indices, :]

	# Calculate the eye width and height for both eyes for the entire batch
	left_eye_width_batch = np.max(left_eye_landmarks_batch[:, :, 0], axis=1) - np.min(left_eye_landmarks_batch[:, :, 0], axis=1)
	left_eye_height_batch = np.max(left_eye_landmarks_batch[:, :, 1], axis=1) - np.min(left_eye_landmarks_batch[:, :, 1], axis=1)

	right_eye_width_batch = np.max(right_eye_landmarks_batch[:, :, 0], axis=1) - np.min(right_eye_landmarks_batch[:, :, 0], axis=1)
	right_eye_height_batch = np.max(right_eye_landmarks_batch[:, :, 1], axis=1) - np.min(right_eye_landmarks_batch[:, :, 1], axis=1)

	# Calculate the eye centers for the entire batch
	left_eye_center_batch = np.mean(left_eye_landmarks_batch, axis=1)
	right_eye_center_batch = np.mean(right_eye_landmarks_batch, axis=1)

	return left_eye_center_batch, right_eye_center_batch, left_eye_width_batch, left_eye_height_batch, right_eye_width_batch, right_eye_height_batch

def get_gaze_position(image, facial_landmarks, facial_landmarks_3dmm, gaze_model):

	gaze_pred, _ = estimate_gaze_direction(image, facial_landmarks, gaze_model)

	facial_landmarks_3dmm = facial_landmarks_3dmm.detach().cpu().numpy() 
	gaze_pred = gaze_pred.detach().cpu().numpy()
	
	# Get the eye width and height
	left_eye_center, right_eye_center, left_eye_width, left_eye_height, right_eye_width, right_eye_height = get_eye_width_height_batch(facial_landmarks_3dmm)
	
	theta_p = gaze_pred[:, 0]
	theta_y = gaze_pred[:, 1]

	# Calculate gaze positions for the left eye
	y_left_batch = -left_eye_height / 2 * np.sin(theta_p)
	x_left_batch = -left_eye_width / 2 * np.sin(theta_y) * np.cos(theta_p)
	x_left_batch = x_left_batch.reshape(-1, 1)
	y_left_batch = y_left_batch.reshape(-1, 1)
	gaze_left_batch = np.hstack((x_left_batch, y_left_batch)) + left_eye_center

	# Calculate gaze positions for the right eye
	y_right_batch = -right_eye_height / 2 * np.sin(theta_p)
	x_right_batch = -right_eye_width / 2 * np.sin(theta_y) * np.cos(theta_p)
	x_right_batch = x_right_batch.reshape(-1, 1)
	y_right_batch = y_right_batch.reshape(-1, 1)
	gaze_right_batch = np.hstack((x_right_batch, y_right_batch)) + right_eye_center

	return gaze_right_batch, gaze_left_batch

def save_image(image, save_image_path, range = (-1, 1)):
			
	grid = torch_utils.save_image(
		image,
		save_image_path,
		normalize=True,
		# range=range,
	)

def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok = True)

" Trasnform torch tensor images from range [-1,1] to [0,255]"
def torch_range_1_to_255(image): 
	img_tmp = image.clone()
	min_val = -1
	max_val = 1
	img_tmp.clamp_(min=min_val, max=max_val)
	img_tmp.add_(-min_val).div_(max_val - min_val + 1e-5)
	img_tmp = img_tmp.mul(255.0)
	return img_tmp


# Function for stickman drawing
def draw_stickman(poses, right_eye = None, left_eye = None):
	image_size = 256; stickman_thickness = 2
	edges_parts  = [
		list(range( 0, 17)), # face
		list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
		list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
		list(range(36, 42)), list(range(42, 48)), # right eye, left eye
		list(range(48, 60)), list(range(60, 68))] # lips

	closed_parts = [
		False, False, False, False, False, True, True, True, True]

	colors_parts = [
		(  255,  255,  255), 
		(  255,    0,    0), (    0,  255,    0),
		(    0,    0,  255), (    0,    0,  255), 
		(  255,    0,  255), (    0,  255,  255),
		(  255,  255,    0), (  255,  255,    0)]
	
	stickmen = [];count = 0
	for pose in poses:
		xy = pose.cpu().numpy() # (68,2)
		xy = xy[None, :, None].astype(np.int32)
		stickman = np.ones((image_size, image_size, 3), np.uint8)
		for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
			stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=stickman_thickness)

		if right_eye is not None:
			left_eye_local = (int(left_eye[count][0]), int(left_eye[count][1]))
			right_eye_local = (int(right_eye[count][0]), int(right_eye[count][1]))
			stickman = cv2.circle(stickman, left_eye_local, 5, (255, 165, 0), -1)  # Yellow circle to mark the gaze position
			stickman = cv2.circle(stickman, right_eye_local, 5, (128, 0, 128), -1)
			
		stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
		stickmen.append(stickman)
		count += 1
	stickmen = torch.stack(stickmen)
	stickmen = (stickmen - 0.5) * 2. 

	return stickmen


