import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from math import log10, sqrt

#展示未添加噪声的图片
def show_images(images_path, image_size):
    images = []
    path = os.path.join(images_path, '*g')
    files = glob.glob(path) #所有文件全路径列表
    for fl in files:
        image = cv2.imread(fl, 0)  # 通过路径把图读进来(0代表灰度图片)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR) #图片重塑,shape是(256, 256)
        #展示图片
        winname = 'Image ' + str(files.index(fl)+1)
        cv2.imshow(winname, image) #用窗口展示图片
        cv2.waitKey(0) #等待一个字符
        cv2.destroyWindow(winname) #销毁用来展示图片的窗口
        images.append(image)  # 将未经过处理的图片添加到列表中
    return images #没有噪声的(256, 256)灰度图片

#为图片添加椒盐噪声并保存
def add_sp_and_save(images):
    sp_noise_imgs = [] #添加完椒盐噪声之后的图片集
    index = 0 #图片索引
    for image in images: #对于images中保存的所有 没有噪声的(256, 256)灰度图片
        sp_noise_img = salt_and_pepper(image, 0.02)
        index += 1
        cv2.imwrite('Image_Salt_and_Pepper_Noise\\' + str(index) + '.jpg', sp_noise_img)
        sp_noise_imgs.append(sp_noise_img)
    return sp_noise_imgs #返回添加椒盐噪声之后的图片集

#定义添加椒盐噪声的函数,src灰度图片,per噪声比例
def salt_and_pepper(src, per):
    sp_noise_img = np.copy(src) #深拷贝
    sp_noise_num = int(per * src.shape[0] * src.shape[1]) #噪点数量
    for i in range(sp_noise_num):
        #生成闭区间[low,high]上离散均匀分布的整数值;若high=None，则取值区间变为[1,low]
        rand_x = np.random.randint(0, src.shape[0])
        rand_y = np.random.randint(0, src.shape[1])
        #随机将一些点变成白(255),或黑(0)
        if np.random.randint(0, 2) == 0:
            sp_noise_img[rand_x, rand_y] = 0
        else:
            sp_noise_img[rand_x, rand_y] = 255
    return sp_noise_img #添加椒盐噪声之后的图片

#对未添加噪声的图片进行数据预处理
def pretrain_images(images):
    images_data = []
    for image in images:
        image = image.astype(np.float32)  # 将图片像素点数据转化成float32类型
        image = np.multiply(image, 1.0 / 255.0)  # 每个像素点值都在0到255之间,进行归一化
        images_data.append(image)
    return images_data

#展示添加椒盐噪声之后的图片,并进行数据预处理
def show_sp_noise_images(sp_noise_imgs):
    sp_data = []
    #展示添加椒盐噪声之后的图片
    index = 0  # 图片索引
    for image in sp_noise_imgs:
        index += 1
        winname = 'Image ' + str(index)
        cv2.imshow(winname, image) #用窗口展示图片
        cv2.waitKey(0) #等待一个字符
        cv2.destroyWindow(winname) #销毁用来展示图片的窗口
        image = image.astype(np.float32)  # 将图片像素点数据转化成float32类型
        image = np.multiply(image, 1.0 / 255.0)  # 每个像素点值都在0到255之间,进行归一化
        sp_data.append(image)
    return sp_data

#展示添加高斯噪声之后的图片,并进行数据预处理
def show_gaussian_noise_images(gaussian_noise_imgs):
    gaussian_data = []
    #展示添加高斯噪声之后的图片
    index = 0 #图片索引
    for image in gaussian_noise_imgs:
        index += 1
        winname = 'Image ' + str(index)
        cv2.imshow(winname, image)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        gaussian_data.append(image)
    return gaussian_data

#定义添加高斯噪声的函数,src灰度图片,scale噪声标准差
def gaussian(src, scale):
    gaussian_noise_img = np.copy(src) #深拷贝
    noise = np.random.normal(0, scale, size=(src.shape[0], src.shape[1])) #噪声
    add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #未经检查的图片
    add_noise_and_check += noise
    add_noise_and_check = add_noise_and_check.astype(np.int16)
    # #原来的错误算法
    # # gaussian_noise_num = int(per * src.shape[0] * src.shape[1])
    # # for i in range(gaussian_noise_num):
    # #     rand_x = np.random.randint(0, src.shape[0])
    # #     rand_y = np.random.randint(0, src.shape[1])
    # #     #添加高斯噪声
    # #     gaussian_noise_img[rand_x, rand_y] += int(10 * np.random.randn()) #要添加的噪声数值
    for i in range(len(add_noise_and_check)):
        for j in range(len(add_noise_and_check[0])):
            if add_noise_and_check[i][j] > 255:
                add_noise_and_check[i][j] = 255
            elif add_noise_and_check[i][j] < 0:
                add_noise_and_check[i][j] = 0
    '''
    uint8是无符号整数,0到255之间
    0黑,255白
    256等价于0,-1等价于255
    每256个数字一循环
    '''
    gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
    return gaussian_noise_img #返回添加了高斯噪声之后的图片

#为图片添加高斯噪声并保存
def add_gaussian_and_save(images):
    gaussian_noise_imgs = [] #添加完高斯噪声之后的图片集
    index = 0 #图片索引
    for image in images:
        gaussian_noise_img = gaussian(image, 20)
        index += 1
        cv2.imwrite('Image_Gaussian_Noise\\' + str(index) + '.jpg', gaussian_noise_img)
        gaussian_noise_imgs.append(gaussian_noise_img)
    return gaussian_noise_imgs #返回添加高斯噪声之后的图片集

#将(9, 256, 256)的数据集切分成8*8的patch块
#原始数据和噪音数据都使用这个方法进行patch块切分
def image_data_patch(data):
    image_data = np.copy(data) #将数据载入
    patch_size = 8 #patch块边长
    #shape=(9, 62001, 8, 8),dtype=np.float32
    patches = np.zeros(shape=(image_data.shape[0],
                              int((image_data.shape[1] - patch_size + 1) ** 2),
                              patch_size,
                              patch_size),
                       dtype=np.float32)
    for image_count in range(len(data)): #所有9张图片
        number = 0 #每一张图片当前patch块数量
        image = data[image_count] #其中一张图片(256, 256)
        for row in range(0, len(image) - 7, 1): #所有行,每1个格标注一次
            for col in range(0, len(image[0]) - 7,  1): #所有列
                patches[image_count][number] = image[row:row + 8, col:col + 8]
                number += 1
    return patches

#QR分解
def householder_reflection(A):
    """Householder变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
    return (Q, R)

#高斯噪声的单层字典去噪
def gaussian_single_layer_dictionarylearning(g_p_d, i_p_d, dl_lambda):
    gaussian_patches_data = np.copy(g_p_d)
    images_patches_data = np.copy(i_p_d)

    index = 0 #图片索引
    print('开始从高斯噪声的图像中提取字典...')

    #使用高斯噪声训练字典
    #每一行的data减去均值除以方差，这是zscore标准化的方法
    gaussian_mean = np.mean(gaussian_patches_data, axis=0) #保存下来
    gaussian_patches_data -= gaussian_mean

    #初始化MiniBatchDictionaryLearning类，并按照初始参数初始化类的属性
    dico = MiniBatchDictionaryLearning(n_components=256, alpha=dl_lambda, n_iter=200)
    V = dico.fit(gaussian_patches_data).components_

    #画出V中的字典，下面逐行解释
    '''figsize方法指明图片的大小，4.2英寸宽，4英寸高。其中一英寸的定义是80个像素点'''
    plt.figure(figsize=(8.2, 8))
    #循环画出100个字典V中的字(n_components是字典的数量)
    '''enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    同时列出数据和数据下标，一般用在 for 循环当中。'''
    for i, comp in enumerate(V[:256]):
        plt.subplot(16, 16, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    #6个参数与注释后的6个属性对应
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)#left, right, bottom, top, wspace, hspace
    plt.show()
    print('dictionary shape : ', V.shape)
    print('Dictionary learned on %d patches' % (len(gaussian_patches_data)))

    print('完成从高斯噪声的图像中提取字典...')



    print('开始高斯噪声的稀疏表示...')

    #复原图片和原图的误差
    differents = []

    #字典表示策略
    transform_algorithms = [
        ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
         {'transform_n_nonzero_coefs': 7})
    ]

    # 清空此文件夹中之前的文件
    remove_files('Image_Gaussian_SingleLayer_DictionaryLearning')
    for title, transform_algorithm, kwargs in transform_algorithms:
        # 通过set_params对第二阶段的参数进行设置
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        # transform根据set_params对设完参数的模型进行字典表示，表示结果放在code中。
        # code总共有100列，每一列对应着V中的一个字典元素，
        # 所谓稀疏性就是code中每一行的大部分元素都是0，这样就可以用尽可能少的字典元素表示回去。
        code = dico.transform(gaussian_patches_data)
        # code矩阵乘V得到复原后的矩阵patches
        #样本(62001, 64) = 稀疏表示(62001, 256) * 过完备字典(256, 64)
        patches = np.dot(code, V)

        #还原数据预处理
        patches += gaussian_mean

        # 将patches从（62001，64）变回（62001，8，8）
        patches = patches.reshape(len(gaussian_patches_data), *(8, 8))

        if transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        # 通过reconstruct_from_patches_2d函数将patches重新拼接回图片
        reconstruction_image = reconstruct_from_patches_2d(patches, (256, 256))

        # 计算复原图片和原图的误差
        psnr_score = psnr(
            reconstruct_from_patches_2d(
                images_patches_data.reshape(len(images_patches_data), *(8, 8)),
                (256, 256)),
            reconstruction_image, PIXEL_MAX=1)

        differents.append(psnr_score)

        plt.figure()
        plt.imshow(reconstruction_image, cmap='gray')
        plt.title('字典表示策略 : ' + title + '\npsnr_score : ' + str(psnr_score))
        plt.show()
        #保存去噪复原图
        index += 1
        cv2.imwrite('Image_Gaussian_SingleLayer_DictionaryLearning\\' +
                    'algorithms_' + str(index) +
                    '_psnr_score_' + str(round(psnr_score, 2)).replace('.', '__') + '.jpg', reconstruction_image * 255)


    print('完成高斯噪声的稀疏表示...')

#高斯噪声的深度字典去噪
def gaussian_deepdictionarylearning(g_p_d, i_p_d, dl_lambda1, dl_lambda2):
    gaussian_patches_data = np.copy(g_p_d)
    images_patches_data = np.copy(i_p_d)
    index = 0  # 图片索引
    gaussian_mean = np.mean(gaussian_patches_data, axis=0)  # 保存下来
    gaussian_patches_data -= gaussian_mean
    dico1 = MiniBatchDictionaryLearning(n_components=144, alpha=dl_lambda1, n_iter=200)
    V1 = dico1.fit(gaussian_patches_data).components_ #(144, 64)
    print('dictionary1 shape : ', V1.shape)
    transform_algorithms = [
        (
            ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
            {'transform_n_nonzero_coefs': 7}),
            ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
            {'transform_n_nonzero_coefs': 7})
        )
    ]
    #title, transform_algorithm, kwargs
    remove_files('Image_Gaussian_DeepDictionaryLearning')
    for layer1, layer2 in transform_algorithms:
        dico1.set_params(transform_algorithm=layer1[1], **layer1[2])
        code1 = dico1.transform(gaussian_patches_data)
        #激活函数
        # code1 = relu_reverse_2(code1)
        # code1 = relu_reverse_1(code1)
        dico2 = MiniBatchDictionaryLearning(n_components=256, alpha=dl_lambda2, n_iter=200)
        V2 = dico2.fit(code1).components_
        print('dictionary2 shape : ', V2.shape)
        dico2.set_params(transform_algorithm=layer2[1], **layer2[2])
        code2 = dico2.transform(code1)
        patches = np.dot(np.dot(code2, V2), V1)
        # patches = np.dot(relu(np.dot(code2, V2)), V1)

        patches += gaussian_mean

        # 将patches从（62001，64）变回（62001，8，8）
        patches = patches.reshape(len(gaussian_patches_data), *(8, 8))

        if layer1[1] == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        # 通过reconstruct_from_patches_2d函数将patches重新拼接回图片
        reconstruction_image = reconstruct_from_patches_2d(patches, (256, 256))

        # 计算复原图片和原图的误差
        psnr_score = psnr(
            reconstruct_from_patches_2d(
                images_patches_data.reshape(len(images_patches_data), *(8, 8)),
                (256, 256)),
            reconstruction_image, PIXEL_MAX=1)

        plt.figure()
        plt.imshow(reconstruction_image, cmap='gray')
        plt.title('字典表示策略 : ' + layer1[0] + '\npsnr_score : ' + str(psnr_score))
        plt.show()
        # 保存去噪复原图
        index += 1
        cv2.imwrite('Image_Gaussian_DeepDictionaryLearning\\' +
                    'algorithms_' + str(index) +
                    '_psnr_score_' + str(round(psnr_score, 2)).replace('.', '__') + '.jpg', reconstruction_image * 255)

#椒盐噪声的单层字典去噪
def sp_single_layer_dictionarylearning(s_p_d, i_p_d, dl_lambda):
    sp_patches_data = np.copy(s_p_d)
    images_patches_data = np.copy(i_p_d)
    index = 0  # 图片索引
    print('开始从椒盐噪声的图像中提取字典...')

    # 使用椒盐噪声训练字典
    # 每一行的data减去均值除以方差，这是zscore标准化的方法
    sp_mean = np.mean(sp_patches_data, axis=0)  # 保存下来
    sp_patches_data -= sp_mean

    # 初始化MiniBatchDictionaryLearning类，并按照初始参数初始化类的属性
    dico = MiniBatchDictionaryLearning(n_components=256, alpha=dl_lambda, n_iter=200)
    V = dico.fit(sp_patches_data).components_

    # 画出V中的字典，下面逐行解释
    '''figsize方法指明图片的大小，4.2英寸宽，4英寸高。其中一英寸的定义是80个像素点'''
    plt.figure(figsize=(8.2, 8))
    # 循环画出100个字典V中的字(n_components是字典的数量)
    '''enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    同时列出数据和数据下标，一般用在 for 循环当中。'''
    for i, comp in enumerate(V[:256]):
        plt.subplot(16, 16, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    # 6个参数与注释后的6个属性对应
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)  # left, right, bottom, top, wspace, hspace
    plt.show()
    print('dictionary shape : ', V.shape)
    print('Dictionary learned on %d patches' % (len(sp_patches_data)))

    print('完成从椒盐噪声的图像中提取字典...')



    print('开始椒盐噪声的稀疏表示...')

    # 复原图片和原图的误差
    differents = []

    # 四种不同的字典表示策略
    transform_algorithms = [
        ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
         {'transform_n_nonzero_coefs': 7})
    ]

    # 清空此文件夹中之前的文件
    remove_files('Image_Salt_and_Pepper_SingleLayer_DictionaryLearning')
    for title, transform_algorithm, kwargs in transform_algorithms:
        # 通过set_params对第二阶段的参数进行设置
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        # transform根据set_params对设完参数的模型进行字典表示，表示结果放在code中。
        # code总共有100列，每一列对应着V中的一个字典元素，
        # 所谓稀疏性就是code中每一行的大部分元素都是0，这样就可以用尽可能少的字典元素表示回去。
        code = dico.transform(sp_patches_data)
        # code矩阵乘V得到复原后的矩阵patches
        patches = np.dot(code, V)

        # 还原数据预处理
        patches += sp_mean

        # 将patches从（62001，64）变回（62001，8，8）
        patches = patches.reshape(len(sp_patches_data), *(8, 8))

        if transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        # 通过reconstruct_from_patches_2d函数将patches重新拼接回图片
        reconstruction_image = reconstruct_from_patches_2d(patches, (256, 256))

        # 计算复原图片和原图的误差
        psnr_score = psnr(
            reconstruct_from_patches_2d(
                images_patches_data.reshape(len(images_patches_data), *(8, 8)),
                (256, 256)),
            reconstruction_image, PIXEL_MAX=1)


        differents.append(psnr_score)

        plt.figure()
        plt.imshow(reconstruction_image, cmap='gray')
        plt.title('字典表示策略 : ' + title + '\npsnr_score : ' + str(psnr_score))
        plt.show()
        # 保存去噪复原图
        index += 1
        cv2.imwrite('Image_Salt_and_Pepper_SingleLayer_DictionaryLearning\\' +
                    'algorithms_' + str(index) +
                    '_psnr_score_' + str(round(psnr_score, 2)).replace('.', '__') + '.jpg', reconstruction_image * 255)

    print('完成椒盐噪声的稀疏表示...')

#椒盐噪声的深度字典去噪
def sp_deepdictionarylearning(s_p_d, i_p_d, dl_lambda1, dl_lambda2):
    sp_patches_data = np.copy(s_p_d)
    images_patches_data = np.copy(i_p_d)
    index = 0  # 图片索引
    sp_mean = np.mean(sp_patches_data, axis=0)  # 保存下来
    sp_patches_data -= sp_mean
    dico1 = MiniBatchDictionaryLearning(n_components=144, alpha=dl_lambda1, n_iter=200)
    V1 = dico1.fit(sp_patches_data).components_  # (144, 64)
    print('dictionary1 shape : ', V1.shape)
    transform_algorithms = [
        (
            ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
             {'transform_n_nonzero_coefs': 7}),
            ('Orthogonal Matching Pursuit\n7 atoms', 'omp',
             {'transform_n_nonzero_coefs': 7})
        )
    ]
    # title, transform_algorithm, kwargs
    remove_files('Image_Salt_and_Pepper_DeepDictionaryLearning')
    for layer1, layer2 in transform_algorithms:
        dico1.set_params(transform_algorithm=layer1[1], **layer1[2])
        code1 = dico1.transform(sp_patches_data)
        #激活函数
        # code1 = sigmoid(code1)
        # code1 = relu_reverse_2(code1)
        dico2 = MiniBatchDictionaryLearning(n_components=256, alpha=dl_lambda2, n_iter=200)
        V2 = dico2.fit(code1).components_
        print('dictionary2 shape : ', V2.shape)
        dico2.set_params(transform_algorithm=layer2[1], **layer2[2])
        code2 = dico2.transform(code1)
        #逆激活函数
        # patches = np.dot(np.dot(code2, V2), V1)
        patches = np.dot(np.dot(code2, V2), V1)

        patches += sp_mean

        # 将patches从（62001，64）变回（62001，8，8）
        patches = patches.reshape(len(sp_patches_data), *(8, 8))

        if layer1[1] == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        # 通过reconstruct_from_patches_2d函数将patches重新拼接回图片
        reconstruction_image = reconstruct_from_patches_2d(patches, (256, 256))

        # 计算复原图片和原图的误差
        psnr_score = psnr(
            reconstruct_from_patches_2d(
                images_patches_data.reshape(len(images_patches_data), *(8, 8)),
                (256, 256)),
            reconstruction_image, PIXEL_MAX=1)


        plt.figure()
        plt.imshow(reconstruction_image, cmap='gray')
        plt.title('字典表示策略 : ' + layer1[0] + '\npsnr_score : ' + str(psnr_score))
        plt.show()
        # 保存去噪复原图
        index += 1
        cv2.imwrite('Image_Salt_and_Pepper_DeepDictionaryLearning\\' +
                    'algorithms_' + str(index) +
                    '_psnr_score_' + str(round(psnr_score, 2)).replace('.', '__') + '.jpg', reconstruction_image * 255)

#清空一个文件夹中所有文件
def remove_files(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

#sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

#sigmoid函数逆函数
def sigmoid_reverse(x):
    s = -np.log(1 / x - 1)
    return s

#relu函数
def relu(x):
    return np.maximum(0, x)

#relu函数逆函数1
#负数和0不变,正数变成1/正数
def relu_reverse_1(x):
    temp_list = []
    for item in x.flat:
        if item > 0:
            item = 1 / item
        else:
            pass
        temp_list.append(item)
    x = np.array(temp_list).reshape((x.shape[0], x.shape[1]))
    return x

#relu函数逆函数2
#负数不变,正数变成1/整数,0加上一个很小的数字之后变成1/很小的数
def relu_reverse_2(x):
    temp_list = []
    for item in x.flat:
        if item > 0:
            item = 1 / item
        elif item == 0:
            item = 999999
        else:
            pass
        temp_list.append(item)
    x = np.array(temp_list).reshape((x.shape[0], x.shape[1]))
    return x

# 计算和原图之间的误差
#要求的shape是(256, 256, 1)或(256, 256, 1)
def cal_diff(noise_img, img):
    return np.sqrt(np.sum((noise_img - img) ** 2))

#psnr评分,PIXEL_MAX是最大像素点
def psnr(img1, img2, PIXEL_MAX=255):
    if PIXEL_MAX == 255:
        mse = np.mean( (img1/1.0 - img2/1.0) ** 2 )
    elif PIXEL_MAX == 1:
        mse = np.mean( (img1/255.0 - img2/255.0) ** 2)
    return 20 * log10(PIXEL_MAX / sqrt(mse))