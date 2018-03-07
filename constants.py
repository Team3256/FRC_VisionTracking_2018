CAM_LOCATION = 0 # will eventually be set to MJPG stream address: http://roborio-3256-frc.local:1181/?action=stream. set to 0 for camera directly connected to TX1 over USB
ROBORIO_IP = '10.32.56.2' # needed for NetworkTables

MODEL_FILENAME = 'model.caffemodel' # filename of the DetectNet model file
PROTO_FILENAME = 'proto.prototxt' # filename of the DetectNet prototxt

SHOW_FRAMES = False # whether or not to display frames using opencv windows
SEND_COORDS = True # whether or not to send coordinates of bboxes over NetworkTables

WIDTH_RES = 320
FOV = 4.7
FOCAL = WIDTH_RES / FOV
