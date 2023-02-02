#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys

MIN_MATCH_COUNT = 8

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 2
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(10 / self._cam_fps)


	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		#TODO run SIFT on the captured frame
		camera_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		original_image = cv.imread(self.template_path,cv.IMREAD_GRAYSCALE)

		# Initiate SIFT detector
		sift = cv.SIFT_create()

		# find the keypoints and descriptors with SIFT
		kpOrig, desOrig = sift.detectAndCompute(original_image,None)
		kpCamera, desCamera = sift.detectAndCompute(camera_image,None)

		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)

		flann = cv.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(desOrig,desCamera,k=2)

		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)

		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kpOrig[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kpCamera[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()
			h,w = original_image.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)
			frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
		else:
			print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
			matchesMask = None

		pixmap = self.convert_cv_to_pixmap(frame)
		self.live_image_label.setPixmap(pixmap)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

	
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
