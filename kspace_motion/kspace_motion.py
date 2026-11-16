import math
import copy
import torch
import warnings
import numpy as np
from collections import defaultdict
from scipy.interpolate import pchip_interpolate
from typing import Dict, Tuple, List, Union, Optional
import os
import sys
# optional local checkout of pytorch-finufft, e.g. export PYTORCH_FINUFFT_PATH=/path/to/pytorch-finufft
if os.environ.get("PYTORCH_FINUFFT_PATH"):
    sys.path.append(os.environ["PYTORCH_FINUFFT_PATH"])
import pytorch_finufft
from torchio.data.subject import Subject
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform, FourierTransform
from .kspace_artifacts import KSpaceContext, apply_kspace_artifacts, signal_level


class RandomMotionGPU(RandomTransform, IntensityTransform):
    r"""Simulate motion artefact

    Simulate motion artefact from a random time course of object positions.

    parameters relatated to simulate 3 types of displacement random noise swllow or sudden mouvement

    Args:
        nT (int): number of points of the simulated time course (not very important, as it will be interpolated afer
            to the select phase axis dimension
        maxDisp (float, float): (min, max) value of displacement in the perlin noise (useless if noiseBasePars is 0)
        maxRot (float, float): (min, max) value of rotation in the perlin noise (useless if noiseBasePars is 0)
        noiseBasePars (float, float): (min, max) base value of the perlin noise to generate for the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        swallowFrequency (int, int): (min, max) number of swallowing movements to generate in the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        swallowMagnitude (float, float): (min, max) magnitude of the swallowing movements to generate
        suddenFrequency (int, int): (min, max) number of sudden movements to generate in the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        suddenMagnitude (float, float): (min, max) magnitude of the sudden movements to generate
            if euler_motion_params is not None previous parameter are not used
        maxGlobalDisp (float, float): (min, max) of the global translations. A random number is taken from
            this interval to scale each translations if they are bigger. If None, it won't be used. Note that the
            previous maxDisp maxRot swallowMagnitude suddenMagnitude are controling the amplitude of the perturbation
            but we then add all of then so that at the end we do not control the global amplitude. In order to find
            the global amplitude (and scaled it), we take the max of all 2 by 2 positions distance
        maxGlobalRot same as  maxGlobalDisp but for Rotations
        preserve_center_frequency_pct : percentage of the total time to be set to zero for simulated euler motion
            parameters. (Usefull to avoid big global displacement)
        euler_motion_params : movement parameters to use (if specified, no movement is simulated). Note that in this
            case this is no more a random transform, and one could use directly MotionFromTimeCourse.
            if you modify this attribute after the class instance has been created, you need to also set the
            attribute simulate_displacement to false
        phase_encoding_choice (tuple of ints): potential phase encoding dims (slowest) to use
            (randomly chosen from the input tuple)
        kspace_artifacts: list of :mod:`kspace_motion.kspace_artifacts` applied to the k-space after the
            motion, before the reconstruction (noise, spikes, Gibbs ...)

    Note for suddenFrequency and swallowFrequency min max must differ and the max is never achieved,
        so to have 0 put (0,1)

    todo define parameter as degrees or translation in RandomMotion (specify chosen random distribution)
    todo short description of time course simu + reference gallichan retromoco

    .. note:: quite long execution times compare to other transforms
    .. warning:: extra dependency on finufft, if rotation is modeled, and SimpleElastix if coregistration
        is chosen

    """

    def __init__(
            self,
            nT: int = 200,
            maxDisp: Tuple[float, float] = (8, 10),#(2,5),
            maxRot: Tuple[float, float] = (8, 10), #(2,5),
            noiseBasePars: Tuple[float, float] = (5,15),
            swallowFrequency: Tuple[float, float] = (1,5),
            swallowMagnitude: Tuple[float, float] = (5,10),
            suddenFrequency: Tuple[int, int] = (1,5),
            suddenMagnitude: Tuple[float, float] = (5,10),
            maxGlobalDisp: Tuple[float, float] = (5,10),
            maxGlobalRot: Tuple[float, float] = (5,10),
            preserve_center_frequency_pct: float = 0,
            euler_motion_params: Union[List, np.ndarray, str] = None,
            phase_encoding_choice: List = [1],
            device="cpu",
            kspace_artifacts: Optional[List] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.kspace_artifacts = kspace_artifacts
        self.nT = nT
        self.maxDisp = maxDisp
        self.maxRot = maxRot
        self.noiseBasePars = noiseBasePars
        self.swallowFrequency = swallowFrequency
        self.swallowMagnitude = swallowMagnitude
        self.suddenFrequency = suddenFrequency
        self.suddenMagnitude = suddenMagnitude
        self.maxGlobalDisp = maxGlobalDisp
        self.maxGlobalRot = maxGlobalRot
        self.preserve_center_frequency_pct = preserve_center_frequency_pct
        self.kspace_order = self._rand_kspace_order(phase_encoding_choice)
        self.device = device
        if euler_motion_params is None:
            self.euler_motion_params = None
            self.simulate_displacement = True
        else:
            if isinstance(euler_motion_params, np.ndarray):
                euler_motion_params = torch.from_numpy(euler_motion_params)
            elif not isinstance(euler_motion_params, torch.Tensor):
                euler_motion_params = torch.as_tensor(np.asarray(euler_motion_params))
            self.euler_motion_params = euler_motion_params.to(device)

            self.simulate_displacement = False


    def apply_transform(self, subject: Subject) -> Subject:

        arguments = defaultdict(dict)
        if self.simulate_displacement:
            self._simulate_random_trajectory()
        for image_name, image_dict in self.get_images_dict(subject).items():
            arguments["euler_motion_params"][image_name] = self.euler_motion_params
            arguments["kspace_order"][image_name] = self.kspace_order
            arguments["device"][image_name] = self.device
            arguments["kspace_artifacts"][image_name] = self.kspace_artifacts

        transform = MotionGPU(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def random_params(self,
                      maxDisp: Tuple[float, float] = (2,5),
                      maxRot: Tuple[float, float] = (2,5),
                      noiseBasePars: Tuple[float, float] = (5,15),
                      swallowFrequency: Tuple[float, float] = (0,5),
                      swallowMagnitude: Tuple[float, float] = (2,6),
                      suddenFrequency: Tuple[int, int] = (0,5),
                      suddenMagnitude: Tuple[float, float] = (2,6),
                      maxGlobalDisp: Tuple[float, float] = None,
                      maxGlobalRot: Tuple[float, float] = None
                      ):
        maxDisp_r = self._rand_uniform(min=maxDisp[0], max=maxDisp[1])
        maxRot_r = self._rand_uniform(min=maxRot[0], max=maxRot[1])
        noiseBasePars_r = self._rand_uniform(min=noiseBasePars[0], max=noiseBasePars[1])
        swallowMagnitude_r = [self._rand_uniform(min=swallowMagnitude[0], max=swallowMagnitude[1]),
                            self._rand_uniform(min=swallowMagnitude[0], max=swallowMagnitude[1])]

        suddenMagnitude_r = [self._rand_uniform(min=suddenMagnitude[0], max=suddenMagnitude[1]),
                           self._rand_uniform(min=suddenMagnitude[0], max=suddenMagnitude[1])]

        swallowFrequency_r = torch.randint(swallowFrequency[0], swallowFrequency[1], (1,)).item()
        suddenFrequency_r = torch.randint(suddenFrequency[0], suddenFrequency[1], (1,)).item()

        maxGlobalDisp_r = self._rand_uniform(min=maxGlobalDisp[0],
                                           max=maxGlobalDisp[1]) if maxGlobalDisp else float('inf')
        maxGlobalRot_r = self._rand_uniform(min=maxGlobalRot[0],
                                          max=maxGlobalRot[1]) if maxGlobalRot else float('inf')

        # prba to include the different type of noise
        proba_noiseBase = noiseBasePars[2] if len(noiseBasePars) == 3 else 1
        proba_swallow = swallowFrequency[2] if len(swallowFrequency) == 3 else 1
        proba_sudden = suddenFrequency[2] if len(suddenFrequency) == 3 else 1

        do_noise, do_swallow, do_sudden = False, False, False

        while (do_noise or do_swallow or do_sudden) is False:  # at least one is not false
            do_noise = self._rand_uniform() <= proba_noiseBase
            do_swallow = self._rand_uniform() <= proba_swallow
            do_sudden = self._rand_uniform() <= proba_sudden
        if do_noise is False: noiseBasePars_r = 0
        if do_swallow is False: swallowFrequency_r = 0
        if do_sudden is False: suddenFrequency_r = 0

        return (maxDisp_r, maxRot_r, noiseBasePars_r, swallowMagnitude_r, suddenMagnitude_r, swallowFrequency_r,
                suddenFrequency_r, maxGlobalDisp_r, maxGlobalRot_r)


    def perlinNoise1D(self, npts, weights):
        if not isinstance(weights, list):
            weights = range(int(round(weights)))
            weights = np.power([2] * len(weights), weights)

        n = len(weights)
        xvals = np.linspace(0, 1, npts)
        total = np.zeros((npts, 1))

        for i in range(n):
            frequency = 2 ** i
            this_npts = round(npts / frequency)

            if this_npts > 1:
                total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts),
                                                        self._rand_uniform(shape=this_npts)[..., np.newaxis],
                                                        xvals)
        #            else:
        # TODO does it matter print("Maxed out at octave {}".format(i))

        total = total - np.min(total)
        total = total / np.max(total)
        return total.reshape(-1)

    def _simulate_random_trajectory(self):
        """
        Simulates the parameters of the transformation through the vector euler_motion_params using 6 dimensions
        (3 translations and 3 rotations).
        """

        maxDisp, maxRot, noiseBasePars, swallowMagnitude, suddenMagnitude, swallowFrequency, suddenFrequency, \
        maxGlobalDisp, maxGlobalRot = self.random_params(maxDisp=self.maxDisp, maxRot=self.maxRot,
                 noiseBasePars=self.noiseBasePars, swallowFrequency=self.swallowFrequency,
                 swallowMagnitude=self.swallowMagnitude, suddenFrequency=self.suddenFrequency,
                 suddenMagnitude=self.suddenMagnitude, maxGlobalDisp=self.maxGlobalDisp,
                 maxGlobalRot=self.maxGlobalRot)

        if noiseBasePars > 0:
            euler_motion_params = np.asarray([self.perlinNoise1D(self.nT, noiseBasePars) - 0.5 for _ in range(6)])
            euler_motion_params[:3] *= maxDisp
            euler_motion_params[3:] *= maxRot
        else:
            euler_motion_params = np.zeros((6, self.nT))
        # add in swallowing-like movements - just to z direction and pitch
        if swallowFrequency > 0:
            swallowTraceBase = np.exp(-np.linspace(0, 100, self.nT))
            swallowTrace = np.zeros(self.nT)

            for i in range(swallowFrequency):
                rand_shifts = int(round(self._rand_uniform() * self.nT))
                rolled = np.roll(swallowTraceBase, rand_shifts, axis=0)
                swallowTrace += rolled

            euler_motion_params[2, :] += swallowMagnitude[0] * swallowTrace
            euler_motion_params[3, :] += swallowMagnitude[1] * swallowTrace

        # add in random sudden movements in any direction
        if suddenFrequency > 0:
            suddenTrace = np.zeros(euler_motion_params.shape)

            for i in range(suddenFrequency):
                iT_sudden = int(np.ceil(self._rand_uniform() * self.nT))
                to_add = np.asarray([suddenMagnitude[0] * (2 * self._rand_uniform(shape=3) - 1),
                                     suddenMagnitude[1] * (2 * self._rand_uniform(shape=3) - 1)]).reshape((-1, 1))
                suddenTrace[:, iT_sudden:] = np.add(suddenTrace[:, iT_sudden:], to_add)

            euler_motion_params += suddenTrace

        if self.preserve_center_frequency_pct:
            center = np.int32(np.floor(euler_motion_params.shape[1] / 2))
            nbpts = np.int32(np.floor(euler_motion_params.shape[1] * self.preserve_center_frequency_pct / 2))
            euler_motion_params[:, center - nbpts:center + nbpts] = 0

        # rescale to global max if asked
        # max is compute for all trans (or rot) diff
        if self.maxGlobalDisp is not None:
            trans_diff = euler_motion_params.T[:,None,:3] - euler_motion_params.T[None,:,:3]  #numpy broadcating rule !
            ddtrans_max = np.linalg.norm(trans_diff, axis=2).max()
            if ddtrans_max > 0:
                euler_motion_params[:3, :] = euler_motion_params[:3, :] * maxGlobalDisp / ddtrans_max
        if self.maxGlobalRot is not None:
            ddrot_max = np.linalg.norm(euler_motion_params.T[:,None,3:] - euler_motion_params.T[None,:,3:] , axis=-1).max()
            if ddrot_max > 0:
                euler_motion_params[3:, :] = euler_motion_params[3:, :] * maxGlobalRot / ddrot_max

        self.euler_motion_params = torch.from_numpy(euler_motion_params).to(self.device)


    def _rand_uniform(self, min=0.0, max=1.0, shape=1):
        rand = torch.FloatTensor(shape).uniform_(min, max)
        if shape == 1:
            return rand.item()
        return rand.numpy()


    def _rand_kspace_order(self, phase_encoding_choise):
        if not isinstance(phase_encoding_choise, list):
            phase_encoding_choise = [phase_encoding_choise]
        #choose phase axis (among user defined choices, ie list of possibilitie)
        random_phase_axis = self._rand_choice(phase_encoding_choise)
        #choose random order
        kspace_order_axis = torch.randperm(3)
        # put the phase at the end
        result = torch.cat([kspace_order_axis[kspace_order_axis!=random_phase_axis],torch.tensor([random_phase_axis])])

        return result.numpy()

    def _rand_choice(self, array):
        chosen_idx = torch.randint(0, len(array), (1, ))
        return array[chosen_idx]


class MotionGPU(IntensityTransform, FourierTransform):
    r"""Add MRI motion artifact (computed in kspace).

        euler_motion_params : movement parameters to use 
        kspace order (3 tuple) dimension ordering of the kspace axis, the last one will then define the slowest varying
            axis, and this is where the motion takes place (2 other dimension are assumed, step wise stationary)
        phase_encoding_choice (tuple of ints): potential phase encoding dims (slowest) to use
            (randomly chosen from the input tuple)
        kspace_artifacts: list of :mod:`kspace_motion.kspace_artifacts` applied to the acquired k-space
            before the reconstruction. Without motion (all parameters zero) the k-space is computed with
            a plain FFT instead of the NUFFT.
        """

    def __init__(
            self,
            euler_motion_params: Union[List, np.ndarray, str],
            kspace_order: int = (0,2,1),
            device: str = "cpu",
            kspace_artifacts: Optional[List] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.kspace_order = kspace_order
        self.device = device
        self.euler_motion_params = euler_motion_params
        self.kspace_artifacts = kspace_artifacts
        self.args_names = ("euler_motion_params", "kspace_order", "device", "kspace_artifacts")

    def apply_transform(self, subject: Subject) -> Subject:
        euler_motion_params = self.euler_motion_params
        kspace_order = tuple(self.kspace_order)
        device = self.device
        kspace_artifacts = self.kspace_artifacts

        for image_name, image_dict in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                euler_motion_params = self.euler_motion_params[image_name]
                kspace_order = tuple(self.kspace_order[image_name])
                device = self.device[image_name]
                kspace_artifacts = self.kspace_artifacts[image_name]

            original_image = torch.squeeze(image_dict['data']).to(device).contiguous()
            euler_motion_params = torch.from_numpy(euler_motion_params) if isinstance(euler_motion_params, np.ndarray) else euler_motion_params
            euler_motion_params = euler_motion_params.to(device)
            phase_encoding_shape = original_image.shape[kspace_order[-1]]
            #TODO test if oversampling_pct we should add zero in euler_motion_params and not interp to oversampled shape
            if euler_motion_params.shape[1] != phase_encoding_shape:
                euler_motion_params = _interpolate_torch(euler_motion_params, 
                                                         len_output=phase_encoding_shape, 
                                                         device=device)

            corrupted_im = self.apply_motion_in_kspace(original_image, euler_motion_params, 
                                                       kspace_order, device=device,
                                                       kspace_artifacts=kspace_artifacts)

            # magnitude
            #corrupted_im = abs(corrupted_im)

            image_dict["data"] = corrupted_im.unsqueeze(0)#[np.newaxis, ...]

        return subject
   
    def _rotate_coordinates_1D_motion(self, euler_motion_params, image_shape, kspace_order, device="cpu"):
        euler_motion_params_local = copy.deepcopy(euler_motion_params)
        off_center = torch.tensor([(x/2 -x//2) * 2 for x in image_shape], device=device)
        
        euler_motion_params_local[:3, :] = euler_motion_params_local[:3, :] - \
                                            torch.repeat_interleave(off_center.unsqueeze(1), 
                                                                    euler_motion_params_local.shape[1], axis=1)
        lin_spaces = [torch.linspace(-0.5, 0.5-1/x, x, dtype=torch.float64, device=device)*2*math.pi for x in image_shape]
        meshgrids = torch.meshgrid(*lin_spaces, indexing="ij")
        grid_out = torch.stack([*meshgrids, torch.zeros(image_shape, device=device)])
        dim_order = np.hstack([np.array(0), np.array(kspace_order)+1]) #first dim is 4 (homogeneous coordinates)
        grid_out = grid_out.permute(tuple(dim_order))
        
        if euler_motion_params_local.shape[1] != grid_out.shape[3]:
                    raise ValueError('dimension issue euler_motion_params and phase dim')

        for nnp in range(euler_motion_params_local.shape[1]):
            aff = get_matrix_from_euler_and_trans(euler_motion_params_local[:,nnp], device=device)
            grid_plane = grid_out[:, :, :, nnp]
            shape_mem = grid_plane.shape
            grid_plane_moved = torch.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
            #grid_plane_moved = np.matmul( grid_plane.reshape(4,shape_mem[1]*shape_mem[2]).T, aff.T).T # r0.T * A.T
            grid_out[:, :, :, nnp] = grid_plane_moved.reshape(shape_mem)

        #tricky to insverse a given transpose
        dim_order_back = tuple([np.argwhere(dim_order==k)[0][0] for k in range(4)])
        grid_out = grid_out.permute(dim_order_back) #go back to original order

        return grid_out.contiguous()


    @staticmethod
    def _fft_kspace(image):
        """Centred k-space of `image`, to be reconstructed like the NUFFT of `_nufft_kspace` (magnitude of
        the shifted inverse FFT). For odd sizes the NUFFT grid is offset by half a sample, so only the
        reconstructed magnitude (not the phase of the k-space) is the same."""
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(image.type(torch.complex64))))

    def _nufft_kspace(self, image, euler_motion_params, kspace_order, device="cpu"):
        grid_out = self._rotate_coordinates_1D_motion(euler_motion_params, image.shape,
                                                      kspace_order, device=device)
        x ,y, z = grid_out[0].flatten(), grid_out[1].flatten(), grid_out[2].flatten()
        grid_rot = torch.stack([x, y, z], axis=0)
        f = pytorch_finufft.functional.finufft_type2(grid_rot.type(torch.float32), image.type(torch.complex64), modeord=0, isign=-1)
        f = f * torch.exp(-1j * grid_out[3].flatten())
        return f.reshape(image.shape)

    def _trans_and_nufft_type2(self, image, euler_motion_params, kspace_order, device="cpu",
                               kspace_artifacts=None):
        if bool(torch.any(euler_motion_params != 0)):
            f = self._nufft_kspace(image, euler_motion_params, kspace_order, device=device)
        else:
            f = self._fft_kspace(image)
        if kspace_artifacts:
            context = KSpaceContext(kspace_order=tuple(int(a) for a in kspace_order),
                                    signal_level=signal_level(image))
            f = apply_kspace_artifacts(f, kspace_artifacts, context)
        iout = torch.abs(torch.fft.ifftshift(torch.fft.ifftn(f)))
        return iout

    def apply_motion_in_kspace(self, image, euler_motion_params, kspace_order, device="cpu",
                               kspace_artifacts=None):
        corrupted_im = self._trans_and_nufft_type2(image, euler_motion_params, kspace_order, device=device,
                                                   kspace_artifacts=kspace_artifacts)
        return corrupted_im


def torch_1d_interp(
    x,
    xp,
    fp,
    left = None,
    right= None,
):
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: 1d sequence of floats. x-coordinates. Must be increasing
        fp: 1d sequence of floats. y-coordinates. Must be same length as xp
        left: Value to return for x < xp[0], default is fp[0]
        right: Value to return for x > xp[-1], default is fp[-1]

    Returns:
        The interpolated values, same shape as x.
    """
    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    answer = torch.where(
        x < xp[0],
        left,
        (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )
    answer = torch.where(x > xp[-1], right, answer)
    return answer

def _interpolate_torch(
            euler_motion_params,
            tr_fpars = None,
            tr_to_interpolate = 2.4,
            len_output = 250, 
            device="cpu"
            ):

        fpars_length = euler_motion_params.shape[1]
        if tr_fpars is None:
            xp = torch.linspace(0, 1, fpars_length, device=device)
            x  = torch.linspace(0, 1, len_output, device=device)
        else:
            xp = torch.arange(fpars_length, device=device)*tr_fpars
            x = torch.arange(len_output, device=device)*tr_to_interpolate
        interpolated_fpars = torch.stack([torch_1d_interp(x=x, xp=xp, fp=fp) 
        for fp in euler_motion_params])
        if xp[-1] < x[-1]:
            diff = x[-1] - xp[-1]
            npt_added = diff/tr_to_interpolate
            #print(f'adding {npt_added:.1f}')
        return interpolated_fpars

def get_matrix_from_euler_and_trans(P, rot_order='yxz', device="cpu"):
    # default choosen as the same default as simpleitk
    rot = torch.deg2rad(P[3:6])
    aff = torch.eye(4, device=device, dtype=torch.float64)
    aff[:3,3] = P[:3]  #translation
    if rot_order=='xyz':
        aff[:3,:3]  = euler2mat(rot[0], rot[1], rot[2], axes='sxyz', device=device)
    elif rot_order=='yxz':
        aff[:3,:3] = euler2mat(rot[1], rot[0], rot[2], axes='syxz', device=device) #strange simpleitk convention of euler ... ?
    else:
        raise NotImplementedError(f'rotation order {rot_order} not implemented')
    return aff


# to avoid, an extra dependency, athough this one is nice work from mathew Brett!
# I copy paste the function from transforms3d.euler import euler2mat
# map axes strings to/from tuples of inner axis, parity, repetition, frame

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler2mat(ai, aj, ak, axes='sxyz', device="cpu"):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = torch.eye(3, device=device)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M
