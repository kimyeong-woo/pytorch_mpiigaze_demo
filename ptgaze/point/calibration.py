import cv2
import numpy as np
from scipy.spatial.distance import cdist
from ..common import Face, FacePartsName
from ..gaze_estimator import GazeEstimator
import math

class Calibration:
    def __init__(self, gaze_estimator: GazeEstimator, screen_width, screen_height):
        self.gaze_estimator = gaze_estimator 

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.eye_vector = np.array([0, 0])

        self.point_a = np.array([screen_width / 10, screen_height / 10])
        self.point_b = np.array([screen_width * 9 / 10, screen_height / 10])
        self.point_c = np.array([screen_width * 9 / 10, screen_height * 9 / 10])
        self.point_d = np.array([screen_width / 10, screen_height * 9 / 10])

        self.scale = np.array([0.0, 0.0])
        self.point = np.array([0.0, 0.0])

    def calc_eye_2d_vector(self, face: Face) -> np.ndarray:
        """
        눈의 각도를 계산합니다.
        """
        eye_angles = np.zeros((2, 3), dtype=np.float32)
        eye_positions = np.zeros((2, 3), dtype=np.float32)
        eye_composition = np.zeros((2, 3), dtype=np.float32)

        for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
            eye = getattr(face, key.name.lower())
            eye_angles[i] = eye.normalized_gaze_vector
            eye_angles[i][0] = -eye_angles[i][0]
            eye_positions[i] = eye.center / eye.distance
            eye_positions[i] = self.gaze_estimator.camera.convert_to_camera_direction(eye_positions[i])
            eye_composition[i] = eye_angles[i] - eye_positions[i]

        cross_product = np.cross(eye_composition[0], eye_composition[1])
        res = cross_product[:2]/-cross_product[2]

        res = np.sum(eye_composition, axis=0)
        res = res[:2] / -res[2]

        return res

    def calculate_filtered_center(self, points):
        """
        튀어있는 점(outliers)을 제거하고 중앙점을 계산합니다.
        """
        # numpy 배열로 변환
        points = np.array(points, dtype=np.float32)

        # 중앙점 계산 (초기값)
        center = np.mean(points, axis=0)

        # 거리 계산 (중앙점과 각 점 사이의 거리)
        distances = cdist(points, [center])

        # 거리의 IQR(사분위 범위) 계산
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr  # 이상치 기준

        # 이상치가 아닌 점들만 필터링
        inliers = points[distances.flatten() <= threshold]

        # 필터링된 점들로 새로운 중앙점 계산
        refined_center = np.mean(inliers, axis=0)

        return np.array([refined_center[0], refined_center[1]])

    def calc_filtered_centers(self, points):
        return [self.calculate_filtered_center(p) for p in points]

    def calc_trs_matrix(self, origins, dests):
        """
        원본 점과 변환된 점을 기반으로 변환 행렬을 계산합니다.
        """
        # numpy 배열로 변환
        origins = np.array(origins, dtype=np.float32)
        dests = np.array(dests, dtype=np.float32)

        # 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(dests, origins)
        return matrix

    def calc_trs_transform(self, matrix, point):
        """
        변환 행렬을 사용하여 점을 변환합니다.
        """
        # numpy 배열로 변환
        transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), matrix)[0][0]
        return np.array([int(transformed_point[0]), int(transformed_point[1])])

    def calc_point(self, point):
        """
        화면 좌표를 계산합니다.
        """
        x = int(point[0] * self.scale[0] + self.point[0])
        y = int(point[1] * self.scale[1] + self.point[1])
        return np.array([x, y])

    def draw_calcd_point(self, image, face):
        """
        화면에 점을 그립니다.
        """
        self.calc_center(face)
        point = self.calc_point(self.eye_vector)
        cv2.circle(image, (point[0], point[1]), 5, (0, 255, 0), -1)
