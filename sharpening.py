#锐化对于深层去噪几乎无作用
#图片在锐化的同时,也会将噪点一起显现出来

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#图片锐化1
def sharpen_zero(img):
    kernel = np.array([[0, -1, 0], [-1, 9.5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(img, -1, kernel=kernel)
    return dst

#图片锐化2
def sharpen_minus1(img):
    kernel = np.array([[-1, -1, -1], [-1, 19, -1], [-1, -1, -1]], np.float32) #锐化
    dst = cv2.filter2D(img, -1, kernel=kernel)
    return dst

#对读入的图片进行数据预处理
def load_image(img):
    image = np.copy(img)
    image = cv2.resize(image, (256, 256), 0, 0, cv2.INTER_LINEAR) #图片重塑,shape是(256, 256)
    image = image.astype(np.float32)  #将图片像素点数据转化成float32类型
    image = np.multiply(image, 1.0 / 255.0)  #每个像素点值都在0到255之间,进行归一化
    return image #返回处理之后的图片

#计算原图和另一张图的误差
def cal_different(img, diff_img):
    image = np.copy(img) #原图
    different_image = np.copy(diff_img) #误差图
    different = np.sqrt(np.sum((image - different_image) ** 2)) #误差
    return different

#显示图片(image要展示的图片,title要显示的题目)
def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()



#原图
path_without_noise = os.path.join('Image_Denoising', 'barbara.jpg')
image = cv2.imread(path_without_noise, 0) #原图(以灰度方式读取)
image = load_image(image) #数据预处理之后的图片
show_image(image,  '原图 diff : ' + str(cal_different(image, image)))
print('原图误差diff : ', cal_different(image, image))



print('开始椒盐噪声锐化...')

#椒盐噪声单层去噪
path_sp_singlelayer = os.path.join(
    'Image_Salt_and_Pepper_SingleLayer_DictionaryLearning',
    'algorithms_1_different_16.jpg')
image_sp_singlelayer = cv2.imread(path_sp_singlelayer, 0)
image_sp_singlelayer = load_image(image_sp_singlelayer)
show_image(image_sp_singlelayer, '椒盐噪声单层去噪 diff : ' +
           str(cal_different(image, image_sp_singlelayer)))
print('椒盐噪声单层去噪误差diff : ', cal_different(image, image_sp_singlelayer))

#椒盐噪声多层去噪
path_sp_deepdictionarylearning = os.path.join(
    'Image_Salt_and_Pepper_DeepDictionaryLearning',
    'algorithms_1_different_13.jpg'
)
image_sp_deepdictionarylearning = cv2.imread(path_sp_deepdictionarylearning, 0)
image_sp_deepdictionarylearning = load_image(image_sp_deepdictionarylearning)
show_image(image_sp_deepdictionarylearning,
           '椒盐噪声深度去噪 diff : ' +
           str(cal_different(image, image_sp_deepdictionarylearning)))
print('椒盐噪声深度去噪误差diff : ',
      cal_different(image, image_sp_deepdictionarylearning))

#椒盐噪声多层去噪之后的图片进行锐化处理
sharp_sp_deepdictionarylearning_1 = sharpen_zero(image_sp_deepdictionarylearning)
#归一化
sharp_sp_deepdictionarylearning_1 -= sharp_sp_deepdictionarylearning_1.min()
sharp_sp_deepdictionarylearning_1 /= sharp_sp_deepdictionarylearning_1.max()
show_image(sharp_sp_deepdictionarylearning_1,
           '经过第一种锐化处理后的椒盐噪声深度去噪图片diff : ' +
           str(cal_different(image, sharp_sp_deepdictionarylearning_1)))
print('使用第一种锐化算法后的椒盐噪声深度去噪误差diff : ',
      cal_different(image, sharp_sp_deepdictionarylearning_1))

sharp_sp_deepdictionarylearning_2 = sharpen_minus1(image_sp_deepdictionarylearning)
#归一化
sharp_sp_deepdictionarylearning_2 -= sharp_sp_deepdictionarylearning_2.min()
sharp_sp_deepdictionarylearning_2 /= sharp_sp_deepdictionarylearning_2.max()
show_image(sharp_sp_deepdictionarylearning_2,
           '经过第二种锐化处理后的椒盐噪声深度去噪图片diff : ' +
           str(cal_different(image, sharp_sp_deepdictionarylearning_2)))
print('使用第二种锐化算法后的椒盐噪声深度去噪误差diff : ',
      cal_different(image, sharp_sp_deepdictionarylearning_2))

print('完成椒盐噪声锐化...')



print('开始高斯噪声锐化...')

#椒盐噪声单层去噪
path_gaussian_singlelayer = os.path.join(
    'Image_Gaussian_SingleLayer_DictionaryLearning',
    'algorithms_1_different_11.jpg')
image_gaussian_singlelayer = cv2.imread(path_gaussian_singlelayer, 0)
image_gaussian_singlelayer = load_image(image_gaussian_singlelayer)
show_image(image_gaussian_singlelayer, '高斯噪声单层去噪 diff : ' +
           str(cal_different(image, image_gaussian_singlelayer)))
print('高斯噪声单层去噪误差diff : ', cal_different(image, image_gaussian_singlelayer))



#椒盐噪声多层去噪
path_gaussian_deepdictionarylearning = os.path.join(
    'Image_Gaussian_DeepDictionaryLearning',
    'algorithms_1_different_13.jpg'
)
image_gaussian_deepdictionarylearning = cv2.imread(path_gaussian_deepdictionarylearning, 0)
image_gaussian_deepdictionarylearning = load_image(image_gaussian_deepdictionarylearning)
show_image(image_gaussian_deepdictionarylearning,
           '高斯噪声深度去噪 diff : ' +
           str(cal_different(image, image_gaussian_deepdictionarylearning)))
print('高斯噪声深度去噪误差diff : ',
      cal_different(image, image_gaussian_deepdictionarylearning))



#椒盐噪声多层去噪之后的图片进行锐化处理
sharp_gaussian_deepdictionarylearning_1 = sharpen_zero(image_gaussian_deepdictionarylearning)
#归一化
sharp_gaussian_deepdictionarylearning_1 -= sharp_gaussian_deepdictionarylearning_1.min()
sharp_gaussian_deepdictionarylearning_1 /= sharp_gaussian_deepdictionarylearning_1.max()
show_image(sharp_gaussian_deepdictionarylearning_1,
           '经过第一种锐化处理后的高斯噪声深度去噪图片diff : ' +
           str(cal_different(image, sharp_gaussian_deepdictionarylearning_1)))
print('使用第一种锐化算法后的高斯噪声深度去噪误差diff : ',
      cal_different(image, sharp_gaussian_deepdictionarylearning_1))



sharp_gaussian_deepdictionarylearning_2 = sharpen_minus1(image_gaussian_deepdictionarylearning)
#归一化
sharp_gaussian_deepdictionarylearning_2 -= sharp_gaussian_deepdictionarylearning_2.min()
sharp_gaussian_deepdictionarylearning_2 /= sharp_gaussian_deepdictionarylearning_2.max()
show_image(sharp_gaussian_deepdictionarylearning_2,
           '经过第二种锐化处理后的高斯噪声深度去噪图片diff : ' +
           str(cal_different(image, sharp_gaussian_deepdictionarylearning_2)))
print('使用第二种锐化算法后的高斯噪声深度去噪误差diff : ',
      cal_different(image, sharp_gaussian_deepdictionarylearning_2))

print('完成高斯噪声锐化...')