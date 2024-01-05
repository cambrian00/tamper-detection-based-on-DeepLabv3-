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


# 1.对模型进行预训练，创建生成器
batch_size = 2
train_path = "data/preTrain"
test_path = "data/test/CASIA1"
activation = "softmax"
dual = True
size = 256
trainGen = trainGenerator(batch_size,train_path,"image","label",dict(),image_color_mode = "rgb",
                    mask_color_mode = "grayscale",num_class = 2,save_to_dir = None,target_size = (size,size),
                    mask_target_size = (512,512),
                    seed = 1,activation=activation,dual = dual)

testGen = trainGenerator(1,test_path,"image","label",dict(),image_color_mode = "rgb",
                    mask_color_mode = "grayscale",num_class = 2,save_to_dir = None,target_size = (size,size),
                    mask_target_size = (512,512),
                    seed = 1,activation=activation,dual = dual)

model = Deeplabv3(weights="pascal_voc", input_tensor=None, input_shape=(size, size, 3), classes=2,
                  backbone='xception',OS=16, alpha=1., activation="softmax", dual=dual)
model.summary()

# 2.模型的训练和编译
def scheduler(epoch):
    if(epoch % 1 == 0 and epoch != 0):
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5), loss=categorical_crossentropy,
              metrics=['accuracy'], loss_weights=[1.0, 1.0])

model.fit_generator(trainGen,steps_per_epoch=62997//16+1, epochs=5, callbacks=[reduce_lr],
                   validation_data=testGen, validation_steps=916)
model.save_weights("../checkpoint/pretrain_weights_256_512.h5")


