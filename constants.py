# Network Tables
ROBORIO_IP = '10.32.56.2' # needed for NetworkTables
SEND_COORDS = False # whether or not to send coordinates of bboxes over NetworkTables

# Camera
CAM_LOCATION = 0
WIDTH_RES = 360
FOV = 1.0
FOCAL = WIDTH_RES / FOV
RATIO_SCALE = (1280.0/720.0) / (360.0/240.0) # needed because processed images are resized to 360x240
SHOULD_RESIZE = True
RESIZE_RES = (240, 160)

# Neural Network
NEURAL_NETWORK = False
MODEL_FILENAME = 'model.caffemodel'
PROTO_FILENAME = 'proto.prototxt'

# OpenCV
SHOW_FRAMES = False

# Stream
STREAM_IP = '10.32.56.179'
STREAM_PORT = 5805


