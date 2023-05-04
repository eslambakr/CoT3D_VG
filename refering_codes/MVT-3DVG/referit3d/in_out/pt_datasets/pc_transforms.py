"""
A small connexion of pointcloud transforms from TorchPoints.
"""
import torch
import random
import numpy as np
from .utils import flipcoin


def euler_angles_to_rotation_matrix(theta, random_order=False):
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]]
    )

    R_y = torch.tensor(
        [[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]]
    )

    R_z = torch.tensor(
        [[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]]
    )

    matrices = [R_x, R_y, R_z]
    if random_order:
        random.shuffle(matrices)
    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


class ChromaticTranslation(object):
    """Add random color to the image, data must contain an rgb attribute between 0 and 1

    Parameters
    ----------
    trans_range_ratio:
        ratio of translation i.e. translation = 2 * ratio * rand(-0.5, 0.5) (default: 1e-1)
    """

    def __init__(self, trans_range_ratio=1e-1, p=0.5):
        self.trans_range_ratio = trans_range_ratio
        self.p = p

    def __call__(self, data):
        if flipcoin(percent=self.p*100):
            tr = (torch.rand(1, 3) - 0.5) * 2 * self.trans_range_ratio
            data[:,3:] = torch.clamp(tr + data[:,3:], 0, 1)
        return data

    def __repr__(self):
        return "{}(trans_range_ratio={})".format(self.__class__.__name__, self.trans_range_ratio)


class RandomSymmetry(object):
    """ Apply a random symmetry transformation on the data

    Parameters
    ----------
    axis: Tuple[bool,bool,bool], optional
        axis along which the symmetry is applied
    """

    def __init__(self, axis=[False, False, False], p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data):
        if flipcoin(percent=self.p*100):
            for i, ax in enumerate(self.axis):
                if ax:
                    if torch.rand(1) < 0.5:
                        c_max = torch.max(data[:,:3][:, i])
                        data[:,:3][:, i] = c_max - data[:,:3][:, i]
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)


class RandomNoise(object):
    """ Simple isotropic additive gaussian noise (Jitter)

    Parameters
    ----------
    sigma:
        Variance of the noise
    clip:
        Maximum amplitude of the noise
    """

    def __init__(self, sigma=0.01, clip=0.05, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, data):
        if flipcoin(percent=self.p*100):
            noise = self.sigma * torch.randn(data[:,:3].shape)
            noise = noise.clamp(-self.clip, self.clip)
            data[:,:3] = data[:,:3] + noise
        return data

    def __repr__(self):
        return "{}(sigma={}, clip={})".format(self.__class__.__name__, self.sigma, self.clip)
    

class ChromaticJitter:
    """ Jitter on the rgb attribute of data

    Parameters
    ----------
    std :
        standard deviation of the Jitter
    """

    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, data):
        if flipcoin(percent=self.p*100):
            noise = torch.randn(data[:,3:].shape[0], 3)
            data[:,3:] += noise * self.std
        return data

    def __repr__(self):
        return "{}(std={})".format(self.__class__.__name__, self.std)


class Random3AxisRotation(object):
    """
    Rotate pointcloud with random angles along x, y, z axis

    The angles should be given `in degrees`.

    Parameters
    -----------
    apply_rotation: bool:
        Whether to apply the rotation
    rot_x: float
        Rotation angle in degrees on x axis
    rot_y: float
        Rotation anglei n degrees on y axis
    rot_z: float
        Rotation angle in degrees on z axis
    """

    def __init__(self, apply_rotation: bool = True, rot_x: float = None, rot_y: float = None, rot_z: float = None, p=0.5):
        self._apply_rotation = apply_rotation
        if apply_rotation:
            if (rot_x is None) and (rot_y is None) and (rot_z is None):
                raise Exception("At least one rot_ should be defined")

        self._rot_x = np.abs(rot_x) if rot_x else 0
        self._rot_y = np.abs(rot_y) if rot_y else 0
        self._rot_z = np.abs(rot_z) if rot_z else 0

        self._degree_angles = [self._rot_x, self._rot_y, self._rot_z]
        self.p = p

    def generate_random_rotation_matrix(self):
        thetas = torch.zeros(3, dtype=torch.float)
        for axis_ind, deg_angle in enumerate(self._degree_angles):
            if deg_angle > 0:
                rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
                rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
                thetas[axis_ind] = rand_radian_angle
        return euler_angles_to_rotation_matrix(thetas, random_order=True)

    def __call__(self, data):
        if self._apply_rotation and flipcoin(percent=self.p*100):
            pos = data[:,:3].float()
            M = self.generate_random_rotation_matrix()
            data[:,:3] = pos @ M.T
        return data

    def __repr__(self):
        return "{}(apply_rotation={}, rot_x={}, rot_y={}, rot_z={})".format(
            self.__class__.__name__, self._apply_rotation, self._rot_x, self._rot_y, self._rot_z
        )
