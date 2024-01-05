from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.losses import categorical_crossentropy
import tensorflow as tf

from metric import f1
from model import Deeplabv3
from dual_data import trainGenerator
#from model.loss import *
import skimage.io as io
from skimage import color
import skimage.transform as trans
from glob import glob
import numpy as np