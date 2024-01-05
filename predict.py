import cv2
import numpy as np

from model import Deeplabv3
import skimage.io as io
from glob import glob

activation = "softmax"
classes = 2
dual = True
test_list = ["CASIA1", "column", "cover"]

test_data = "CASIA1"
model = Deeplabv3(input_shape=(512, 512, 3), classes=classes, activation=activation, backbone='xception',
                  dual=dual)
model.load_weights("first.h5")
for i in test_list:
    test_data = i
    print("====================================")
    predict_image_list = glob("data/test/" + test_data + "/image/*")
    print(len(predict_image_list))
    for i in predict_image_list:
        img = io.imread(i)
           # if(img.shape[-1] != 3):
           #     print(i + '  ' + str(img.shape))
           #     continue
           # img = img[:,:,:3]
        w, h, c = img.shape
        img = img / 255.
        img = cv2.resize(img, (512, 512))
        img = np.reshape(img, (1,) + img.shape)
        results, edge = model.predict(img)
        results = np.argmax(results.squeeze(), -1)
        edge = np.argmax(edge.squeeze(), -1)

        results = np.array(results, dtype='uint8')
        edge = np.array(edge, dtype='uint8')

        results = cv2.resize(results, (h, w), interpolation=cv2.INTER_NEAREST)
        edge = cv2.resize(edge, (h, w), interpolation=cv2.INTER_NEAREST)

        save_path = i.replace("../data/test/" + test_data + "/image/", "predict/" + test_data + "/mask/")
        #        print(save_path)
        edge_path = i.replace("../data/test/" + test_data + "/image/", "predict/" + test_data + "/edge/")

        cv2.imwrite(save_path.replace(".tif", ".png"), results * 255)
        cv2.imwrite(edge_path.replace(".tif", ".png"), edge * 255)