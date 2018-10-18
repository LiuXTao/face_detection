from PIL import Image
from face_detection.visualization_utils import show_boxes
from face_detection.detector import detect_face
import cv2
import numpy as np


if __name__ == '__main__':
    img = Image.open('images/test.jpg')
    bounding_boxes, landmarks = detect_face(img, thresholds=[0.6, 0.7, 0.85])

    show_boxes(img, bounding_boxes, landmarks).show()