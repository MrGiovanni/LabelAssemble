import random
from PIL import Image
import numpy as np
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): probability
    """
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 
            img_[mask == 2] = 0 
            return Image.fromarray(img_.astype('uint8')).convert('RGB') 
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
            img[img > 255] = 255                       
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
