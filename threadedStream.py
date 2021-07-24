import cv2
import threading
class StreamGet():
    def __init__(self, src=0):

        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, 300)
        self.stream.set(4, 300)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        thr = threading.Thread(target=self.get, args=())
        thr.start()

        return self
    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
