import threading
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
import socket
import os
import io
# from PIL import Image
import subprocess
from struct import unpack

if os.path.exists("/tmp/socket_test.s"):
  os.remove("/tmp/socket_test.s")    

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind("/tmp/socket_test.s")

server.listen(1)

client_socket, address = server.accept()

image = bytearray(b'')
isRunning = False

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    global isRunning
                    global image
                    if (isRunning == False):
                        isRunning = True
                        bs = client_socket.recv(8)
                        (length,) = unpack('>Q', bs)
                        image = b''
                        while len(image) < length:
                            to_read = length - len(image)
                            image += client_socket.recv(
                                4096 if to_read > 4096 else to_read)
                        print("frame with " + str(len(image)) + " bytes received")
                        isRunning = False
                    self.wfile.write("--jpgboundary")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(image)))
                    self.end_headers()
                    self.wfile.write(image)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://10.32.56.44:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global image
    try:
        server = ThreadedHTTPServer(('10.32.56.44', 8080), CamHandler)
        print "server started"
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()

if __name__ == '__main__':
    main()

