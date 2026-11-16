import torch
import warnings
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
from torchreg import AffineRegistration
from torchio import Subject, Image, DATA, AFFINE
from torchio.data.io import read_image
from torchio.transforms import SpatialTransform


class Coregister(SpatialTransform):
    r"""Apply a coregistration transform to the data
    to a specified target using torchreg (differentiable affine registration).

    The registration is done in the voxel grid (torchreg works in normalised
    voxel coordinates and ignores the physical affines), so the moving images
    and the target are expected to be on the same voxel grid. The estimated
    transform is a free affine (translation, a 3x3 matrix initialised to the
    identity, no zoom/shear by default), it is not constrained to be rigid.

        Args:
            target: target represents the fixed image to coregister to.
            Can be one of:
                    - string: the transform will look for the corresponding
                      key in the subject.
                    - path: the transform will read the data specified in
                      the path.
                    - instance of :class:`~torchio.Image`:
                    the transform will directly use it as a reference.
                    - array or tensor of shape (X, Y, Z) or (C, X, Y, Z)
            In all cases only the first channel of the target is used.
            estimation_mapping: mapping of which coregistration parameters
                                to apply to which image. Can be one of:
                                - string:  the estimated coregistration
                                  parameters of the corresponding key
                                  of the subject will be used for all
                                  data.
                                - dict: the transform will use the keys for
                                  the parameter estimation and the list of
                                  values for application. For example,
                                  ``{'t1': ['t1', 't2', 'mask']}`` will apply
                                  the transform estimated on ``'t1'->'target'``
                                  for the three keys ``'t1'``, ``'t2'`` and
                                  ``'mask'``.
                                - ``None``: the coregistrations are
                                  estimated and applied separately
                                  for each data.
            registration_kwargs: keyword arguments given to
                                 :class:`torchreg.AffineRegistration`
                                 (``scales``, ``iterations``, ``learning_rate``,
                                 ``dissimilarity_function``, ``with_zoom`` ...).
                                 The default dissimilarity is the mean squared
                                 error, so only use it for images with the same
                                 contrast (a callable ``f(moved, static)``
                                 returning a differentiable scalar is expected).
            device: device on which the registration is computed.
            kwargs: See :class:`~torchio.transforms.Transform` for
                    additional keyword arguments.
        """

    def __init__(
            self,
            target: Union[str, Path, Image, np.ndarray, torch.Tensor],
            estimation_mapping: Union[str, Dict, None] = None,
            registration_kwargs: Optional[Dict] = None,
            device: str = "cpu",
            **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.estimation_mapping = estimation_mapping
        self.registration_kwargs = {} if registration_kwargs is None else dict(registration_kwargs)
        self.device = device
        self.estimated_parameter_maps = {}
        if isinstance(self.target, str):
            self.exclude = [*(self.exclude or []), self.target]
        self.args_names = (
            'target',
            'estimation_mapping',
            'registration_kwargs',
            'device',
        )

    def apply_transform(self, subject: Subject) -> Subject:
        # Parse self.target, image_reference is a (X, Y, Z) tensor
        if isinstance(self.target, (Path, str)):
            if self.target in subject.keys():
                image_reference = subject[self.target][DATA][0]
                affine = subject[self.target][AFFINE]
            else:
                try:
                    data, affine = read_image(self.target)
                    image_reference = torch.as_tensor(data)[0]
                except FileNotFoundError:
                    warnings.warn(
                        f'Coregistration target: {self.target} not found for '
                        f'subject: {subject}.\nSkipping coregistration.')
                    return subject
        elif isinstance(self.target, Image):
            image_reference = self.target[DATA][0]
            affine = self.target[AFFINE]
        elif isinstance(self.target, (np.ndarray, torch.Tensor)):
            affine = np.eye(4)
            image_reference = torch.as_tensor(self.target)
            if image_reference.ndim == 4:
                image_reference = image_reference[0]
            if image_reference.ndim != 3:
                warnings.warn(f'Target array must have shape (X, Y, Z) or (C, X, Y, Z), '
                              f'got {tuple(image_reference.shape)}.\nSkipping coregistration.')
                return subject
        else:
            warnings.warn(f'Unrecognized target argument: {self.target}.'
                          f'\nSkipping coregistration.')
            return subject

        imgs = subject.get_images_dict(intensity_only=False,
                                       exclude=self.exclude)
        transform_map = {}
        estimation_mapping = {img_key: img_key for img_key in imgs.keys()}
        if isinstance(self.estimation_mapping, str):
            if self.estimation_mapping in subject.keys():
                estimation_mapping = {
                    self.estimation_mapping:
                    list(imgs.keys())}
            else:
                warnings.warn(f'Key: {self.estimation_mapping} '
                              f'not found in subject: {subject}.'
                              f'\nResuming coregistration without '
                              f'transform map')

        elif isinstance(self.estimation_mapping, Dict):
            estimation_mapping = {
                img_key: estimated_img_key
                for img_key, estimated_img_key
                in self.estimation_mapping.items()
                if img_key in imgs.keys()
            }

        for estimation_img_name, img_names in estimation_mapping.items():

            estimation_img = imgs[estimation_img_name]

            if estimation_img_name not in transform_map.keys():
                registration = self._estimate_coregisteration(
                    estimation_img,
                    image_reference,
                    registration_kwargs=self.registration_kwargs,
                    device=self.device)
                transform_map[estimation_img_name] = registration
            if isinstance(img_names, str):
                img_names = [img_names]
            for img_name in img_names:
                moving_img = imgs[img_name]
                coregistered = self._apply_coregistration(
                    moving_img,
                    transform_map[estimation_img_name],
                    shape=image_reference.shape,
                    device=self.device
                )
                moving_img.set_data(coregistered)
                moving_img.affine = affine
        self.estimated_parameter_maps = transform_map
        return subject

    @staticmethod
    def _estimate_coregisteration(
            img_src: Image,
            img_ref: torch.Tensor,
            registration_kwargs: Optional[Dict] = None,
            device: str = "cpu"
    ) -> AffineRegistration:
        """Estimate the affine registering the first channel of img_src to img_ref"""
        kwargs = dict(is_3d=True, with_zoom=False, with_shear=False, verbose=False)
        kwargs.update(registration_kwargs or {})
        moving = img_src[DATA][0].float()[None, None].to(device)
        static = img_ref.float()[None, None].to(device)
        registration = AffineRegistration(**kwargs)
        registration(moving, static, return_moved=False)
        return registration

    @staticmethod
    def _apply_coregistration(
            img_src: Image,
            registration: AffineRegistration,
            shape,
            device: str = "cpu"
    ) -> torch.Tensor:
        """Resample every channel of img_src with the estimated affine, on the target grid"""
        data = img_src[DATA].float()[None].to(device)  # (1, C, X, Y, Z)
        with torch.no_grad():
            moved = registration.transform(data, tuple(shape))
        return moved[0].cpu()
