import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import random
from scipy.special import comb
from copy import deepcopy
import torch

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,:,h1:h2,w1:w2] = a_trg[:,:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    if L == 0:
        return src_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

def fourier_augmentation_reverse(data, fda_beta=0.1):
    data_np = data.cpu().numpy()
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)#0.05~0.15
    lowf_batch = data_np[::-1]
    fda_data = FDA_source_to_target_np(data_np, lowf_batch, L=this_fda_beta)
    fda_data = torch.tensor(fda_data.astype(data_np.dtype), device=data.device)
    return fda_data

class FDATransform(BasicTransform):
    def get_parameters(self, **data_dict):
        return {'fda_beta': 0.1}
    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['fda_data'] = self._apply_to_image(data_dict['image'], **params)
        return data_dict
    def _apply_to_image(self, image, **params):
        fda_beta = params.get('fda_beta', 0.1)
        return fourier_augmentation_reverse(image, fda_beta=fda_beta)

class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)#0.9~1.1
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        inputs_20 = np.percentile(inputs, slide_limit)
        inputs_80 = np.percentile(inputs, 100 - slide_limit)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image=self.non_linear_transformation(image, inverse=False)
        image=self.location_scale_transformation(image)
        return image

    def Local_Location_Scale_Augmentation(self,image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)

        # match the channel between image and mask
        # mask = np.tile(mask, (1, 3, 1, 1))
        mask = np.tile(mask, np.array(image.shape) // np.array(mask.shape))

        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image

class SLTransform(BasicTransform):
    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None and data_dict.get('segmentation') is not None:
            data_dict['gla_data'], data_dict['lla_data'] = self._apply_to_image_and_seg(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict
    def _apply_to_image_and_seg(self, image, seg, **params):
        image_np = image.cpu().numpy()
        if isinstance(seg, list):
            seg = seg[0]
        seg_np = seg.cpu().numpy()
        image_np_min = image_np.min()
        image_np_max = image_np.max()
        image_np = (image_np - image_np_min) / (image_np_max - image_np_min + 1e-6)
        location_scale = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        gla_data = location_scale.Global_Location_Scale_Augmentation(deepcopy(image_np))
        lla_data = location_scale.Local_Location_Scale_Augmentation(deepcopy(image_np), deepcopy(seg_np))
        gla_data = gla_data * (image_np_max - image_np_min + 1e-6) + image_np_min
        lla_data = lla_data * (image_np_max - image_np_min + 1e-6) + image_np_min
        gla_data = torch.tensor(gla_data.astype(image_np.dtype), device=image.device)
        lla_data = torch.tensor(lla_data.astype(image_np.dtype), device=image.device)
        return gla_data, lla_data
    

class KeyTransform(BasicTransform):
    def apply(self, data_dict, **params):
        if data_dict.get('gla_data') is not None and data_dict.get('lla_data') is not None:
            data_dict['image'] = self._apply_to_slaug(data_dict['gla_data'], data_dict['lla_data'], **params)
        return data_dict
    def _apply_to_slaug(self, gla_data, lla_data, **params):
        image = torch.cat([gla_data, lla_data], dim=0)
        return image