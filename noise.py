import random
from PIL import Image
import numpy as np
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
