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

# 1.创建生成器，正式训练
batch_size = 4
train_path = "data/preTrain"
trainNumber = len(glob(train_path+"/image/*"))
test_path = "data/test/CASIA1"
testNumber = len(glob(test_path + "/image/*"))
activation = "softmax"
dual = True
size = 512

trainGen = trainGenerator(batch_size,train_path,"image","label",dict(),image_color_mode = "rgb",
                    mask_color_mode = "grayscale",num_class = 1,save_to_dir = None,target_size = (size,size),
                          mask_target_size = (512,512),
                    seed = 1,activation=activation,dual = dual)

testGen = trainGenerator(1,test_path,"image","label",dict(),image_color_mode = "rgb",
                    mask_color_mode = "grayscale",num_class = 1,save_to_dir = None,target_size = (size,size),
                         mask_target_size = (512,512),
                    seed = 1,activation=activation,dual = dual)

model = Deeplabv3(weights="pascal_voc", input_tensor=None, input_shape=(size,size, 3), classes=2,
                  backbone='xception',OS=16, alpha=1., activation=activation,dual = dual)
# model.load_weights("deeplabV2.h5")
model.summary()


def scheduler(epoch):
    if(epoch % 2 == 0 and epoch != 0):
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)
modelCheck = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.h5",
                monitor='val_loss', verbose=0,
                save_best_only=False, save_weights_only=True,
                mode='auto', period=1)

#binary_focal_loss(gamma=2.0, alpha=0.25)  categorical_crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4,decay=1e-5),loss = categorical_crossentropy,
                metrics  = ['accuracy'],loss_weights=[1.0, 1.0])

model.fit_generator(trainGen,steps_per_epoch=trainNumber//batch_size+1,epochs=10,callbacks=[reduce_lr,modelCheck],
                   validation_data=testGen,validation_steps=testNumber)
model.save_weights("first.h5")

