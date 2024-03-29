#!/usr/bin/python
import threading
import constants
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
import socket
import os
import io
import cv2
from PIL import Image
import subprocess
from struct import unpack
import constants
from networktables import NetworkTables
import logging
logging.basicConfig(level=logging.DEBUG)

import argparse
import math

from google.protobuf import text_format
import numpy as np
import scipy.misc

cam = cv2.VideoCapture(constants.CAM_LOCATION)

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk - currently unused

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores

def getAngle(xcord):
    distanceFromCenter = (((constants.WIDTH_RES - 1) / 2) - xcord) * constants.RATIO_SCALE
    return (math.atan(distanceFromCenter / constants.FOCAL) * 180) / math.pi

def classify(image, net, transformer):
    _, channels, height, width = transformer.inputs['data']
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = scipy.misc.imresize(image, (height, width), 'bilinear')
    images = [pil_image]
    
    # Classify the image
    scores = forward_pass(images, net, transformer)
    
    ### Process the results

    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81
    imcv = None
    biggestAngle = 0
    biggestWidth = 0
    biggestBoxCords = []
    for i, image_results in enumerate(scores):
        print '==> Image #%d' % i
        imcv = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        points_flat = []
        angle = 0
        for left, top, right, bottom, confidence in image_results:
            if confidence == 0:
                continue
            
            left_i = int(round(left))
            top_i = int(round(top))
            right_i = int(round(right))
            bottom_i = int(round(bottom))
            
            print 'Detected object at [(%d, %d), (%d, %d)] with "confidence" %f' % (
                left_i,
                top_i,
                right_i,
                bottom_i,
                confidence,
            )

            # Get center cube position for angle detection
            centerX = int((left + right) / 2)
            #centerY = int((top + bottom) / 2)        

            #cv2.circle(imcv, (centerX, centerY), 5, (0,0,255), -1)

            # Draw bounding box on screen and get angle
            angle = getAngle(centerX)
            size = cv2.getTextSize(str(getAngle(centerX)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(imcv, str(getAngle(centerX)), (int((left + right) / 2) - size[0][0] / 2, int((top + bottom) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(imcv, (left, top), (right, bottom), (0, 0, 255), 2)

            # Check to see if width is greater, then set this as the greatest if true
            width = right_i - left_i
            if (width > biggestWidth):
                biggestWidth = width
                biggestBoxCords = [left_i, top_i, right_i, bottom_i]
                biggestAngle = angle
            
            if constants.SEND_COORDS:
                points_flat += [left_i, top_i, right_i, bottom_i]

        print("\n----Detection Summary----")
        print("Angle found for biggest cube: " + str(biggestAngle))
        print("Width of biggest cube: " + str(biggestWidth))
        print("Box cords: " + str(biggestBoxCords))

        if constants.SEND_COORDS:
            NetworkTables.getTable('SmartDashboard').putNumberArray('cube_bboxes', biggestBoxCords)
            NetworkTables.getTable('SmartDashboard').putNumber('cube_angle', biggestAngle)
        
        
    return imcv

class CamHandler(BaseHTTPRequestHandler):
    # Load neural network when server starts
    net = get_net(constants.MODEL_FILENAME, constants.PROTO_FILENAME)
    transformer = get_transformer(constants.PROTO_FILENAME)

    def do_GET(self):
        if self.path.endswith('.mjpg') or self.path.endswith('.mjpeg'):
            print("Requested to connect to camera")
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    _, image = cam.read()
                    # Resize input from the camera and run neural network
                    if constants.NEURAL_NETWORK or NetworkTables.getTable('SmartDashboard').getBoolean('neural_network', False):
                        imcv = classify(image, self.net, self.transformer)
                    imcv = cv2.resize(imcv, constants.RESIZE_RES)
                    imgRGB = cv2.cvtColor(imcv,cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = io.BytesIO()
                    jpg.save(tmpFile, 'JPEG')
                    imageSize = len(tmpFile.getvalue())
                    print("Sending image that is " + str(imageSize) + " bytes")
                    if constants.SHOW_FRAMES:
                        cv2.imshow('image', imcv)
                        cv2.waitKey(1)  
                    self.wfile.write("--jpgboundary")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(imageSize))
                    self.end_headers()
                    self.wfile.write(tmpFile.getvalue())
                    tmpFile.close()
                    print("Image sent.")
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://192.168.7.87:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    try:
        if constants.SEND_COORDS:
            NetworkTables.initialize(server=constants.ROBORIO_IP)
        # Start HTTP server
        server = ThreadedHTTPServer((constants.STREAM_IP, constants.STREAM_PORT), CamHandler)
        print "Camera server started successfully"
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()

if __name__ == '__main__':
    main()

