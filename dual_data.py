from __future__ import print_function
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np 
import os
import glob
import skimage.io as io
from skimage import color
import skimage.transform as trans
import cv2
#%%

def adjustData(img,mask,flag_multi_class,num_class,activation=None,dual = False):
    img = img / 255.
    mask = mask / 255.
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    
    if dual == True:
        edge = np.empty(mask.shape)
        for i in range(mask.shape[0]):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 十字形结构
#            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 十字形结构
            erosion = cv2.erode(mask[i,:,:,0], kernel) 
            dilation = cv2.dilate(mask[i,:,:,0], kernel)
            dst = dilation - erosion
#            dst = mask[i,:,:,0] - erosion
            edge[i,:,:,0] = dst
    
    if(activation == "softmax"):
        mask = to_categorical(mask,2)
        if dual == True:
            edge = to_categorical(edge,2)
    if dual == True:
        return (img,[mask,edge])
    else:
        return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),
                   mask_target_size = (512,512),
                    seed = 1,activation=None,dual = False):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = mask_target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class,activation,dual = dual)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i))
        img = img / 255.
        img = trans.resize(img,target_size)
#        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def labelVisualize(num_class,img):
    img = np.argmax(img.squeeze(), -1)
    return img
    #img = img[:,:,0] if len(img.shape) == 3 else img
    #img_out = np.zeros(img.shape + (3,))
    #for i in range(num_class):
        #img_out[img == i,:] = color_dict[i]
    #return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)