# base path to YOLO directory
MODEL_PATH = "videoManager/social_distancing_detector/yolo-coco"
# MODEL_PATH = "yolo-coco"

# base path for mask detection
FACE_MODEL_PATH = "videoManager/social_distancing_detector/face_mask"
# FACE_MODEL_PATH = "face_mask"

# initialize minimum probability to filter weak detections along with the
# threshold when applying non-maxim suppression
MIN_CONF = 0.3
MIN_CONF_FACE = 0.4
NMS_THRESH = 0.3

# should NVIDIA CUDA GPU be used?
USE_GPU = True

# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 300