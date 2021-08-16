from skimage import color
import os
import cv2
import matplotlib.pyplot as plt
from PIL import ImageOps
import numpy as np
import imreg_dft as ird
from scipy.stats import gaussian_kde
import copy


def create_mask(image):
    '''
    This method returns mask of the whole slide image to separate the tissue from the background.
    image: wsi at low resolution level
    '''
    t, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return np.invert(mask)/255


def find_matching_levels(downsamples_s, downsamples_t):
    '''
    :param downsamples_s:
    :param downsamples_t:
    :return:
    '''

    if len(downsamples_s) >= len(downsamples_t):
        list1 = list(downsamples_s)
        list2 = list(downsamples_t)
    else:
        list1 = list(downsamples_t)
        list2 = list(downsamples_s)

    list1_copy = list1[:]
    index_1 = []
    index_2 = []
    for i, e in enumerate(list2):
        if e <= 64:
            elem = min(list1_copy, key=lambda x: abs(x - e))
            index_1.append(i)
            index_2.append(list1.index(elem))
            list1_copy.remove(elem)

    index_2.reverse()
    index_1.reverse()

    if len(downsamples_s) >= len(downsamples_t):
        levels_ref = index_2
        levels_tar = index_1
    else:
        levels_ref = index_1
        levels_tar = index_2

    return levels_ref, levels_tar


def transform_patch_inverse(point, size, transformation, ds_l_s, ds_l_t):
    '''
    Applies transformation to coordinates of reference patch to compute the corresponding vertices in the target image.
    To be able to extract the patch, the bounding box of the four vertices is computed.

    :param point: coordinates of the patch in the reference image
    :param size: patch size
    :param transformation: dictionary with the transformation parameters
    :param ds_l_s: downsampling factor source
    :param ds_l_t: downsampling factor target
    :return: the coordinated of the upper left corner point of the bounding box patch the resulting patch size
    and the coordinates of the vertices.
    '''

    # resize patch for level 0
    size = (size[0] * ds_l_s[0], size[1] * ds_l_s[1])

    #transformation to homogeneous coordinates
    e1 = np.concatenate((point, [1])).T
    e2 = np.concatenate((point + np.array([0, size[1]]), [1])).T
    e3 = np.concatenate((point + np.array([size[0], size[1]]), [1])).T
    e4 = np.concatenate((point + np.array([size[0], 0]), [1])).T

    R = np.linalg.inv(transformation[:,:2])
    T = transformation[:,2:]
    s1 = np.dot(R, (e1[:2]-T.T).T)
    s2 = np.dot(R, (e2[:2]-T.T).T)
    s3 = np.dot(R, (e3[:2]-T.T).T)
    s4 = np.dot(R, (e4[:2]-T.T).T)

    points = np.array([s1[0:2], s2[0:2], s3[0:2], s4[0:2]])

    x_coordinates, y_coordinates = zip(*points)
    start_point = (int(np.around(min(x_coordinates))), int(np.around(min(y_coordinates))))
    patch_size = (int(np.around((max(x_coordinates) - min(x_coordinates)) / ds_l_t[0])),
                  int(np.around((max(y_coordinates) - min(y_coordinates)) / ds_l_t[1])))
    return start_point, patch_size


def get_Random_Patches(file_source, file_target, level_idx, size, initial_transformation, num_samples=1,
                       registration=True, show=False):
    '''
    Generates random samples from the reference image and applies the transformation to find the corresponding patch
    in the target image.

    :param file_target: File with Image that should be registered to the reference image
    :param file_source: File with the image that the traget image is registered to
    :param level_idx: Resolution Level on which the patch samples are generated
    :param size: size of the random patches
    :param initial_transformation: initiale transformation include scaling, rotation and translation
    :param num_samples: number of random patches that should be generated
    :param show: if true the random patches are shown
    :return: random patches from the reference image and the corresponding patch from the target image
    '''
    level_s = level_idx['source']
    level_t = level_idx['target']

    thumbnail = np.array(ImageOps.grayscale(file_source.get_thumbnail((500, 500))))
    dims_thumbnail = thumbnail.shape
    mask = create_mask(thumbnail)

    samples_s = []
    samples_t = []

    offsets = []
    scales = []
    angles = []

    ds = file_source.level_dimensions[0][0] / dims_thumbnail[1]

    ds_l_s = (file_source.level_dimensions[0][0] / file_source.level_dimensions[level_s][0], file_source.level_dimensions[0][1] / file_source.level_dimensions[level_s][1])
    ds_l_t = (file_target.level_dimensions[0][0] / file_target.level_dimensions[level_t][0], file_target.level_dimensions[0][1] / file_target.level_dimensions[level_t][1])

    for i in range(num_samples):
        threshold = 0.85
        tissue = False
        iter = 1
        while not tissue:

            step_ds = int(size[0] /
                          (ds / file_source.level_downsamples[level_s]))
            x_ds = np.random.randint(0, dims_thumbnail[1] - step_ds)
            y_ds = np.random.randint(0, dims_thumbnail[0] - step_ds)
            coverage_value = np.sum(mask[y_ds:y_ds + step_ds, x_ds:x_ds + step_ds])

            if iter > 25:
                threshold -= 0.05
                iter = 1

            # check whether at least 85% of patch are covered with tissue
            if coverage_value > threshold * step_ds * step_ds:
                tissue = True
                x = int(x_ds * ds)
                y = int(y_ds * ds)

                # reference patch
                patch_s = np.uint8(np.array(file_source.read_region(location=(x, y),
                                                                    size=size, level=level_s))[:, :, 0:3])
                samples_s.append(patch_s)

                xy_transformed, patch_size = transform_patch_inverse(point=np.array([x, y]), size=size,
                                                                     transformation=initial_transformation,
                                                                     ds_l_s=ds_l_s, ds_l_t=ds_l_t)

                # target patch with transformation
                patch_t_trans = np.uint8(np.array(file_target.read_region(location=xy_transformed,
                                                                          size=patch_size, level=level_t))[:, :, 0:3])

                # rotate and scale image
                scaled_transform = copy.deepcopy(initial_transformation)
                scaled_transform[0, 0] *= (ds_l_t[0] / ds_l_s[0])
                scaled_transform[1, 1] *= (ds_l_t[1] / ds_l_s[1])
                c_x = int(patch_t_trans.shape[0] / 2)
                c_y = int(patch_t_trans.shape[1] / 2)

                homogeneous = np.array([0, 0, 1])
                M = np.vstack((np.hstack((scaled_transform[:, :2], [[0], [0]])), homogeneous))
                T_c1 = np.vstack((np.array([[1, 0, -c_x], [0, 1, -c_y]]), homogeneous))
                center_t = M @ [c_x, c_y, 1]
                T_c2 = np.vstack((np.array([[1, 0, abs(center_t[0])], [0, 1, abs(center_t[1])]]), homogeneous))
                final_M = T_c2 @ (M @ T_c1)

                shape = tuple((patch_s.shape[0], patch_s.shape[1]))
                final_patch = cv2.warpAffine(patch_t_trans, final_M[0:2, :], shape)

                samples_t.append(final_patch)

                if show:
                    # patch without translation (for comparison)
                    patch_t = np.uint8(
                        np.array(file_target.read_region(location=(x, y), size=size, level=level_t))[:, :, 0:3])

                    plt.subplot(131)
                    plt.imshow(patch_s, cmap="gray")
                    plt.title("source patch")
                    plt.subplot(132)
                    plt.imshow(patch_t, cmap="gray")
                    plt.title("target patch without transformation")
                    plt.subplot(133)
                    plt.imshow(final_patch, cmap="gray")
                    plt.title("Final target patch after transformation")
                    plt.show()

                if registration:
                    try:
                        result_fft = FFT_registration(color.rgb2gray(patch_s), color.rgb2gray(final_patch), show=False)
                        if np.abs(result_fft["angle"]) < 1:
                            offsets.append(result_fft["tvec"])
                            scales.append(result_fft["scale"])
                            angles.append(result_fft["angle"])
                        else:
                            print("patch_rejected")
                    except:
                        print("FFT failes")

            iter += 1
    # if the patches are registered again, the transformation parameters are returned.
    # If not the corresponding patches are returned.
    if registration:
        return np.array(offsets), np.array(angles), np.array(scales)
    return samples_s, samples_t


def FFT_registration(image_source, image_target, show=False):
    '''
    :param image_source: image/patch from one scanner.
    :param image_target: image/patch from the other scanner that should be registered to the reference image.
    :param show: if true the result of the registration is shown.
    :return: transformed image image_target and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in degrees), and translation vector.
    dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
        ``success`` and ``timg`` (the transformed subject image)
    '''

    result = ird.similarity(image_source, image_target, numiter=8)
    # transform from (rows, cols) to (x, y)
    tvector = [result["tvec"][1], result["tvec"][0]]
    angle = result["angle"]
    scale = result["scale"]

    transformation = dict({'tvec': tvector, 'scale': scale, 'angle': angle})

    print("(x,y)-translation [pixel]: {}, angle [°]: {}, scale is {} and success rate {:.4g}"
          .format(tuple(tvector), angle, scale,  result["success"]))

    if show:
        assert "timg" in result
        if os.environ.get("IMSHOW", "yes") == "yes":
            ird.imshow(image_source, image_target, result['timg'], cmap='gray')
            plt.show()
    return transformation


def inital_registration(slide_source, slide_target, downsampling_factor):
    '''
    :param slide_source: whole file scanned by scanner1
    :param slide_target: whole file scanned by the other scanner and that should registered to the reference file
    :param level_idx: the level on with the inital registration should be done, works best on level 7&8
    :return: transformation parameters: isotropic scale factor, rotation angle (in degrees), and translation vector.
    '''

    downsampling_factor = downsampling_factor
    print("inital registration of whole slide")
    # Load images at the lowest level
    img_r = np.array(ImageOps.grayscale(slide_source.get_thumbnail([slide_source.dimensions[0] // downsampling_factor,
                                                                    slide_source.dimensions[1] // downsampling_factor])))
    img_t = np.array(ImageOps.grayscale(slide_target.get_thumbnail([slide_target.dimensions[0] // downsampling_factor,
                                                                    slide_target.dimensions[1] // downsampling_factor])))
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    kpsA_ori, descsA = detector.detectAndCompute(img_r, None)
    kpsB_ori, descsB = detector.detectAndCompute(img_t, None)
    matches = matcher.knnMatch(descsA, descsB, k=2)
    mkp1, mkp2, good = [], [], []
    for match in matches:
        if len(match) < 2:
            break

        m, n = match
        if m.distance < n.distance * 0.6:
            good.append([m])
            mkp1.append(np.array(kpsA_ori[m.queryIdx].pt))
            mkp2.append(np.array(kpsB_ori[m.trainIdx].pt))
    ptsA, ptsB = [], []

    for ptA, ptB in zip(mkp1, mkp2):
        ptA = ptA * (slide_source.dimensions[0] / img_r.shape[1], slide_source.dimensions[1] / img_r.shape[0])
        ptB = ptB * (slide_target.dimensions[0]/img_t.shape[1], slide_target.dimensions[1]/img_t.shape[0])
        ptsA.append(ptA)
        ptsB.append(ptB)

    (E, status) = cv2.estimateAffine2D(np.float32(ptsB), np.float32(ptsA), method=cv2.RANSAC, ransacReprojThreshold=3,
                                       confidence=0.86)
    return E


def wsi_registration(slide_source, slide_target, downsampling_factor):
    '''
    Registration process
    :param slide_source: reference slide for the registration
    :param slide_target: slide that is registered to the reference slide
    :param
    :return: the final transformatin parameters: isotropic scale factor, rotation angle (in degrees), and translation vector.
    '''

    levels_ref, levels_tar = find_matching_levels(slide_source.level_downsamples, slide_target.level_downsamples)

    patch_sizes = []
    size = int((slide_target.level_dimensions[levels_tar[0]][0] + slide_target.level_dimensions[levels_tar[0]][1])/2 * 0.15)
    patch_sizes.append((size, size))

    for i in range(len(levels_ref)-1):
        t = patch_sizes[i][0] + 100
        patch_sizes.append((t, t))

    print("patch sizes: " + str(patch_sizes))

    # inital registration
    initial_transformation = inital_registration(slide_source, slide_target, downsampling_factor=downsampling_factor)

    offset_x = []
    offset_y = []
    weights_offset = []
    angles = []
    scales = []
    adaptive_transformation = copy.deepcopy(initial_transformation)
    previous_offset = [0, 0]

    for l in range(len(levels_ref)):
        levels = dict({'source': levels_ref[l], 'target': levels_tar[l]})
        print("source level: " + str(levels_ref[l]) + " & target level: " + str(levels_tar[l]))

        # register random test patches
        t, a, s = get_Random_Patches(slide_source, slide_target, level_idx=levels, size=patch_sizes[l],
                                     initial_transformation=adaptive_transformation,
                                     num_samples=20, show=False)

        angles.append(a)
        scales.append(s)
        data = np.vstack((np.array(t)[:, 0], np.array(t)[:, 1], a, s))

        try:
            # calculate KDE weights
            kde_weights_offset = gaussian_kde(data, bw_method='scott')(data)
            top_five = np.argsort(kde_weights_offset)[-10:]
            weights = kde_weights_offset[top_five]/kde_weights_offset[top_five].sum()
            # normalize weights
            kde_weights_normed = np.nan_to_num((kde_weights_offset - np.min(kde_weights_offset)) /
                                               (np.max(kde_weights_offset) - np.min(kde_weights_offset)))

            offset_x.append(np.divide(previous_offset[0], slide_source.level_downsamples[levels_ref[l]]) + t[:, 0])
            offset_y.append(np.divide(previous_offset[1], slide_source.level_downsamples[levels_ref[l]]) + t[:, 1])
            weights_offset.append(kde_weights_normed)

            xshift = np.sum(t[:, 0][top_five].T * weights)
            yshift = np.sum(t[:, 1][top_five].T * weights)
            previous_offset += np.multiply(slide_source.level_downsamples[levels_ref[l]], [xshift, yshift])

            adaptive_transformation[:, 2] += np.multiply(slide_source.level_downsamples[levels_ref[l]], [xshift, yshift])
        except:
            return initial_transformation, adaptive_transformation

    return initial_transformation, adaptive_transformation

