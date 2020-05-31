# for the lbp
from skimage import feature

# Classifier
from sklearn.svm import LinearSVC

# to save and load, the model that is created from the classification
from sklearn.externals import joblib #ImportError: cannot import name 'joblib' from 'sklearn.externals'

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2