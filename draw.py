import cv2
import numpy as np
import softmax
import tensorflow as tf

class DrawingWindow():

	def __init__(self, drawHandler):
		# mouse callback function
		self.drawHandler = drawHandler
		self.drawing = False
		self.img = np.zeros((28,28,1), np.uint8)

	def draw_circle(self, event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			ix,iy = x,y

		elif event == cv2.EVENT_MOUSEMOVE:
			if self.drawing == True:
				cv2.circle(self.img, (x,y), 1, (255), -1)

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			self.drawHandler(self.img)

	# creating a drawing 
	def display(self):
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', self.draw_circle)

		while(1):
			cv2.imshow('image', self.img)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('d'):
				self.img[:,:] = 0
			if k == 27:
				break

		cv2.destroyAllWindows()
