import dataset
import numpy as np
import matplotlib.pyplot as plt
'''
导入MiniBatchDictionaryLearning，MiniBatch是字典学习的一种方法，
这种方法专门应用于大数据情况下字典学习。
当数据量非常大时，严格对待每一个样本就会消耗大量的时间，
而MiniBatch通过降低计算精度来换取时间利益，但是仍然能够通过大量的数据学到合理的词典。
换言之，普通的DictionaryLearning做的是精品店，量少而精，但是价格高。
'''
from sklearn.decomposition import MiniBatchDictionaryLearning
'''
导入图片复原函数reconstruct_from_patches_2d，它可以通过pitch复原一整张图片。
'''
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
'''
导入测试工具nose下的异常抛出函数SkipTest
'''
from sklearn.utils.testing import SkipTest
'''
导入SciPy版本检测函数sp_version用于检测版本高低，版本低于0.12的SciPy没有我们需要的样本测试用例
'''
from sklearn.utils.fixes import sp_version

#检测SciPy版本，如果版本太低就抛出一个异常。程序运行结束
if sp_version < (0, 12):
    raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
                   "thus does not include the scipy.misc.face() image.")
#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#要进行去噪的图片索引
denoising_image_index = 0


print('开始数据预处理...')

#读取并展示未加噪声的图片
path_images_without_noise = 'Image_Denoising'
images = dataset.show_images(path_images_without_noise, 256)

#为图片添加椒盐噪声并保存
sp_noise_imgs = dataset.add_sp_and_save(images)

#展示添加椒盐噪声之后的图片,并返回数据预处理之后的图片
#sp_data中的每张图片shape:(256, 256)
#经过了归一化:数据值在0到1之间
sp_data = dataset.show_sp_noise_images(sp_noise_imgs)
sp_data = np.array(sp_data) #(9, 256, 256)
sp_patches = dataset.image_data_patch(sp_data) #图片格式的patch块shape(9, 62001, 8, 8)
#取出第一张椒盐图片的所有patch (62001, 8, 8), reshape成适合训练的形状(62001, 64)
sp_patches_data = sp_patches[denoising_image_index].reshape((62001, 8 * 8)) #适合训练的shape(62001, 64)

#为图片添加高斯噪声并保存
gaussian_noise_imgs = dataset.add_gaussian_and_save(images)

#展示添加高斯噪声之后的图片,并返回数据预处理之后的图片
#gaussian_data:(256, 256)
#经过了归一化:数据值在0到1之间
gaussian_data = dataset.show_gaussian_noise_images(gaussian_noise_imgs)
gaussian_data = np.array(gaussian_data) #(9, 256, 256)
gaussian_patches = dataset.image_data_patch(gaussian_data) #图片格式的patch块shape(9, 62001, 8, 8)
#取出第一张高斯图片的所有patch (62001, 8, 8), reshape成适合训练的形状(62001, 64)
gaussian_patches_data = gaussian_patches[denoising_image_index].reshape((62001, 8 * 8)) #适合训练的shape(62001, 64)

#对原图进行数据预处理
images_data = dataset.pretrain_images(images)
images_data = np.array(images_data) #(9, 256, 256)
images_patches = dataset.image_data_patch(images_data) #图片格式的patch块shape(9, 62001, 8, 8)
#取出第一张原始图片的所有patch (62001, 8, 8), reshape成适合训练的形状(62001, 64)
images_patches_data = images_patches[denoising_image_index].reshape((62001, 8 * 8)) #适合训练的shape(62001, 64)

print('完成数据预处理...')



print('开始展示图片...')

#看看原图的一张图片是什么样的
plt.figure()
plt.imshow(images_data[denoising_image_index], cmap='gray')
plt.title('原图')
plt.show()
print(images_data[denoising_image_index].shape) #看看图片的形状

#看看添加高斯噪声之后的一张图片是什么样的
plt.figure()
plt.imshow(gaussian_data[denoising_image_index], cmap='gray')
gaussian_psnr = dataset.psnr(gaussian_data[denoising_image_index], images_data[denoising_image_index])
plt.title('添加了高斯噪声的图像\npsnr : ' + str(round(gaussian_psnr, 2)))
plt.show()
print(gaussian_data[denoising_image_index].shape) #看看图片的形状

#看看添加椒盐噪声之后的一张图片是什么样的
plt.figure()
plt.imshow(sp_data[denoising_image_index], cmap='gray')
sp_psnr = dataset.psnr(sp_data[denoising_image_index], images_data[denoising_image_index])
plt.title('添加了椒盐噪声的图像\npsnr : ' + str(round(sp_psnr, 2)))
plt.show()
print(sp_data[denoising_image_index].shape) #看看图片的形状

print('完成展示图片...')



print('开始单层字典学习...')

#针对高斯噪声的单层字典学习去噪
dataset.gaussian_single_layer_dictionarylearning(gaussian_patches_data, images_patches_data, 0.1)

#针对椒盐噪声的单层字典学习去噪
dataset.sp_single_layer_dictionarylearning(sp_patches_data, images_patches_data, 0.1)

print('完成单层字典学习...')



print('开始深度字典学习...')

#针对高斯噪声的深层字典学习去噪
dataset.gaussian_deepdictionarylearning(gaussian_patches_data, images_patches_data, 0.01, 0.1)

#针对椒盐噪声的深度字典学习去噪
dataset.sp_deepdictionarylearning(sp_patches_data, images_patches_data, 0.01, 0.1)

print('完成深度字典学习...')