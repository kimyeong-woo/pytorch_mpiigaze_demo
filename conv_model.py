import torch
import numpy as np
import ai_edge_torch
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.common import FacePartsName
from omegaconf import OmegaConf
import cv2
import time

config = OmegaConf.load('./ptgaze/data/configs/mpiigaze.yaml')
config.gaze_estimator.camera_params = './ptgaze/data/calib/camera_params.yaml'
config.gaze_estimator.normalized_camera_params = './ptgaze/data/normalized_camera_params/mpiigaze.yaml'
config.face_detector.mode = 'mediapipe'
config.gaze_estimator.checkpoint = '/Users/jhpark/.ptgaze/models/mpiigaze_resnet_preact.pth'

gaze_estimator = GazeEstimator(config)

image = cv2.imread('./calib_images/sample_img.png')

undistorted = cv2.undistort(
            image, gaze_estimator.camera.camera_matrix,
            gaze_estimator.camera.dist_coefficients)

faces = gaze_estimator.detect_faces(undistorted)
gaze_estimator.estimate_gaze(undistorted, faces[0])

face = faces[0]
images = []
head_poses = []
for key in gaze_estimator.EYE_KEYS:
    eye = getattr(face, key.name.lower())
    image = eye.normalized_image
    normalized_head_pose = eye.normalized_head_rot2d
    if key == FacePartsName.REYE:
        image = image[:, ::-1].copy()
        normalized_head_pose *= np.array([1, -1])
    image = gaze_estimator._transform(image)
    images.append(image)
    head_poses.append(normalized_head_pose)
images = torch.stack(images)
head_poses = np.array(head_poses).astype(np.float32)
head_poses = torch.from_numpy(head_poses)

device = torch.device(gaze_estimator._config.device)
images = images.to(device)
head_poses = head_poses.to(device)

sample_data = (images, head_poses)

edge_model = ai_edge_torch.convert(
    gaze_estimator._gaze_estimation_model,
    sample_data
    )

edge_model.export("/Users/jhpark/.ptgaze/models/mpiigaze.tflite")

current_time = time.time()
gaze_estimator.estimate_gaze(undistorted, faces[0])
print(f"Time taken for gaze estimation: {time.time() - current_time:.4f} seconds")

print(faces[0].reye.normalized_gaze_angles)
print(faces[0].leye.normalized_gaze_angles)

current_time = time.time()
predictions = edge_model(images, head_poses)
print(f"Time taken for edge model gaze estimation: {time.time() - current_time:.4f} seconds")

for i, key in enumerate(gaze_estimator.EYE_KEYS):
    eye = getattr(face, key.name.lower())
    eye.normalized_gaze_angles = predictions[i]
    if key == FacePartsName.REYE:
        eye.normalized_gaze_angles *= np.array([1, -1])
    eye.angle_to_vector()
    eye.denormalize_gaze_vector()
print(faces[0].reye.normalized_gaze_angles)
print(faces[0].leye.normalized_gaze_angles)