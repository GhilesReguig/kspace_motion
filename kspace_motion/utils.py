import itk
import SimpleITK as sitk
import numpy as np
from torchio.data.io import ensure_4d, check_uint_to_int, get_sitk_metadata_from_ras_affine

# Matrices used to switch between LPS and RAS
FLIPXY_33 = np.diag([-1, -1, 1])
FLIPXY_44 = np.diag([-1, -1, 1, 1])


def nib_to_itk(data, affine, force_3d=False, force_4d=False):
    if data.ndim != 4:
        shape = tuple(data.shape)
        raise ValueError(f'Input must be 4D, but has shape {shape}')
    
    array = np.asarray(data)
    affine = np.asarray(affine).astype(np.float64)

    is_multichannel = array.shape[0] > 1 and not force_4d
    is_2d = array.shape[3] == 1 and not force_3d
    if is_2d:
        array = array[..., 0]
    if not is_multichannel and not force_4d:
        array = array[0]
    array = np.ascontiguousarray(array.transpose())  # (D, H, W, C) or (D, H, W), itk memory order

    origin, spacing, direction = get_sitk_metadata_from_ras_affine(
        affine,
        is_2d=is_2d,
    )
    image = itk.GetImageFromArray(array, is_vector=is_multichannel)
    image.SetOrigin(tuple(origin))  # should I add a 4th value if force_4d?
    image.SetSpacing(tuple(spacing))
    direction = np.asarray(direction, dtype=np.float64).reshape((image.GetImageDimension(), -1))
    image.SetDirection(itk.GetMatrixFromArray(direction))

    if data.ndim == 4:
        assert image.GetNumberOfComponentsPerPixel() == data.shape[0]
    return image
                                   

def itk_to_nib(image, keepdim=False, input_spatial_dims=3):
    data = itk.GetArrayFromImage(image).transpose()
    data = check_uint_to_int(data)
    num_components = image.GetNumberOfComponentsPerPixel()
    if num_components == 1:
        data = data[np.newaxis]  # add channels dimension
    #input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 2:
        data = data[..., np.newaxis]
    elif input_spatial_dims == 4:  # probably a bad NIfTI (1, sx, sy, sz, c)
        # Try to fix it
        num_components = data.shape[-1]
        data = data[0]
        data = data.transpose(3, 0, 1, 2)
        input_spatial_dims = 3
    if not keepdim:
        data = ensure_4d(data, num_spatial_dims=input_spatial_dims).numpy()
    assert data.shape[0] == num_components
    affine = get_ras_affine_from_itk(image)
    return data, affine

def convert_sitk_to_itk(sitk_img):
    
    img_array = sitk.GetArrayFromImage(sitk_img)
    itk_img = itk.GetImageFromArray(img_array)
    itk_img.SetOrigin(tuple(sitk_img.GetOrigin()))
    itk_img.SetSpacing(tuple(sitk_img.GetSpacing()))
    itk_img.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_img.GetDirection()), [sitk_img.GetDimension()]*2)))

    return itk_img

def get_ras_affine_from_itk(
    itk_object: itk.Image,
) -> np.ndarray:
    spacing = np.array(itk_object.GetSpacing())
    direction_lps = np.array(itk_object.GetDirection()).reshape(-1)
    origin_lps = np.array(itk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    rotation_ras = np.dot(FLIPXY_33, rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(FLIPXY_33, origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine