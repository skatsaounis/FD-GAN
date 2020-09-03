from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomSizedEarser(object):
    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.5):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1, 1.0)
        W = img.size[0]
        H = img.size[1]
        area = H * W

        if p1 > self.p:
            return img
        else:
            gen = True
            while gen:
                Se = random.uniform(self.sl, self.sh)*area
                re = random.uniform(self.asratio, 1/self.asratio)
                He = np.sqrt(Se*re)
                We = np.sqrt(Se/re)
                xe = random.uniform(0, W-We)
                ye = random.uniform(0, H-He)
                if xe+We <= W and ye+He <= H and xe>0 and ye>0:
                    x1 = int(np.ceil(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1+We))
                    y2 = int(np.floor(y1+He))
                    part1 = img.crop((x1, y1, x2, y2))
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = Image.new('RGB', part1.size, (Rc, Gc, Bc))
                    img.paste(I, part1.size)
                    return img


class RandomColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, p=0.1):
        # Chances of applying effects
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(0, 1)

        if p1 > self.p:
            return img
        else:
            # Draw from uniform to decide which action to take
            action_prob = random.uniform(0, 0.75)

            if action_prob <= 0.25:
                brightness = random.uniform(0.1, 0.2)
                brightness = self._check_input(brightness, 'brightness')
                brightness_factor = random.uniform(brightness[0], brightness[1])
                img = self.adjust_brightness(img, brightness_factor)

            elif action_prob <= 0.5:
                contrast = random.uniform(0.1, 0.2)
                contrast = self._check_input(contrast, 'contrast')
                contrast_factor = random.uniform(contrast[0], contrast[1])
                img = self.adjust_contrast(img, contrast_factor)

            else:
                saturation = random.uniform(0, 0.1)
                saturation = self._check_input(saturation, 'saturation')
                saturation_factor = random.uniform(saturation[0], saturation[1])
                img = self.adjust_saturation(img, saturation_factor)

            #else:
            #    hue = random.uniform(0, 0.01)
            #    hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
            #    hue_factor = random.uniform(hue[0], hue[1])
            #    img = self.adjust_hue(img, hue_factor)

            return img

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None

        return value

    @staticmethod
    def adjust_brightness(img, brightness_factor):
        """Adjust brightness of an Image.
        """
        #img_name = random.uniform(0, 100)
        #img.save(f'augmented/00000000_{img_name}.jpg')
        width, height = img.size
        degenerate = np.zeros((height, width, 3), dtype=np.uint8)
        img = degenerate * (1.0 - brightness_factor) + np.asarray(img) * brightness_factor
        img = Image.fromarray(np.uint8(img))
        #img.save(f'augmented/00000000_{img_name}_aug.jpg')
        return img

    @staticmethod
    def adjust_contrast(img, contrast_factor):
        """Adjust contrast of an Image.
        """
        #img_name = random.uniform(0, 100)
        #img.save(f'augmented/00000000_{img_name}.jpg')
        width, height = img.size
        mean = int(cv2.cvtColor(np.asarray(img).astype(np.uint8), cv2.COLOR_RGB2GRAY).mean() + 0.5)
        degenerate = np.full((height, width, 3), mean, dtype=np.uint8)

        img = degenerate * (1.0 - contrast_factor) + np.asarray(img) * contrast_factor
        img = Image.fromarray(np.uint8(img))
        #img.save(f'augmented/00000000_{img_name}_aug.jpg')
        return img

    @staticmethod
    def adjust_saturation(img, saturation_factor):
        """Adjust color saturation of an image.
        """
        #img_name = random.uniform(0, 100)
        #img.save(f'augmented/00000000_{img_name}.jpg')
        width, height = img.size
        degenerate = cv2.cvtColor(np.asarray(img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        degenerate = np.expand_dims(degenerate, 2)
        img = degenerate * (1.0 - saturation_factor) + np.asarray(img) * saturation_factor
        img = Image.fromarray(np.uint8(img))
        #img.save(f'augmented/00000000_{img_name}_aug.jpg')
        return img

    @staticmethod
    def adjust_hue(img, hue_factor):
        """Adjust hue of an image.
        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.
        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.
        """
        #img_name = random.uniform(0, 100)
        #img.save(f'augmented/00000000_{img_name}.jpg')
        width, height = img.size
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

        img = cv2.cvtColor(np.asarray(img).astype(np.uint8), code=cv2.COLOR_RGB2HSV)

        h = np.asarray(img)[:, :, 0].astype(np.int16)
        img[:, :, 0] = ((h + hue_factor * 180) % 181).astype(np.uint8)
        img = Image.fromarray(np.uint8(img))
        #img.save(f'augmented/00000000_{img_name}_aug.jpg')
        return img
