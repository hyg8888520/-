# 图像质量指标相关代码
# 赛题文件中文部分已替换为拼音

from matplotlib import pyplot as plt
from skimage.exposure import exposure
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
import numpy as np

imgdir = 'fujian1/image_original'
imgdird = []

def colorhist(img):
    plt.figure()
    plt.title('Color Histogram')
    plt.xlabel('level')
    plt.ylabel('number of pixels')
    colors = ('b', 'g', 'r')
    for i, item in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=item)
        plt.xlim([0, 256])

    plt.show()

def color_moments(filename):
    img = filename
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature

def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        filename = filename
        imgdird.append(filename)
read_path(imgdir)


for i in range(9950):
    img1 = cv2.imread('fujian1/image_original' + '\\' + imgdird[i]) # 原图
    img2 = cv2.imread('fujian1/image_message' + '\\' + imgdird[i]) # 嵌入图
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    p = compare_psnr(img1, img2)
    s = compare_ssim(img1, img2, multichannel=True)
    m = compare_mse(img1, img2)

    print('PSNR：{}，SSIM：{}，MSE：{}'.format(p, s, m))

    resimg = cv2.absdiff(img1, img2)
    resimg = cv2.cvtColor(resimg, cv2.COLOR_BGR2GRAY)
    # resimg = np.uint8(np.clip((cv2.add(1.2 * resimg, 70)), 0, 255))
    err_gamma = exposure.adjust_gamma(resimg, 0.67)
    # plt.hist(img1_g.ravel(), 256)
    # plt.hist(img2_g.ravel(), 256)
    # plt.show()

    cv2.imshow('差别图', err_gamma)
    # cv2.imwrite('fujian1/save' + '\\' + str(imgdird[i]) , err_gamma)
    #cv2.imwrite('fujian2/gray' + '\\' + str(imgdird[i]) , img1_g)
    print(imgdird[i])
    # cv2.imwrite('fujian1/gray/me' + '\\' + str(imgdird[i]) , img2_g)
    cv2.waitKey(0)

    # colorhist(img1)
    # colorhist(img2)
    # print(color_moments(img1))
    # print(color_moments(img2))



