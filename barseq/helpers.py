import os
import numpy as np
import re
import scipy
import cv2
from PIL import Image

import scipy as scp
from scipy import ndimage
from scipy.spatial.distance import cdist
import skimage
from skimage.registration import optical_flow_tvl1
import cv2
from pointpats import PointPattern
from pointpats.centrography import ellipse
import tifffile as tif
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation as R
from csbdeep.utils import Path, normalize

import helpers


def point_cloud_parameters(point_cloud):
    pp = PointPattern(point_cloud)
    sx, sy, theta = ellipse(pp.points)
    return sx, sy

def ORB_reg(reference, toalign, nfeatures = 100000, data_type = 'int16', view_matches = False, top_matches = 0.5, return_warped=False):

    '''
    This function uses ORB to register images and correct for the shift between sequencing cycles. Other solutions exist,
    this seemed to be the most straight forward for me. The method does not perform well if not enough key points are selected.
    In my experience this has to be at least 100000.
    :param reference: 2D array -> 2D image that serves as a reference for alignment
    :param toalign: 2D array -> 2D image to be aligned
    :param nfeatures: number of features that ORB uses. Increasing the number helps with the overlap detection
    :param data_type: type of data for final array. not clear to me yet if uint8 is sufficient or should use float 32 or 64.
    :return: 2D array -> 2D image aligned to reference
    '''

    # ORB is finicky and wants 3D arrays so adding another dimension to the input arrays. Also, ORB is double finicky and
    # preferes unsigned ints. I could be wrong.

    ref = np.empty((reference.shape[0], reference.shape[1], 1), np.uint8)
    align = np.empty((toalign.shape[0], toalign.shape[1], 1), np.uint8)
    ref[:, :, 0] = reference
    align[:, :, 0] = toalign

    # Create ORB detector with n features. If problematic, try increasing the number of features.
    orb_detector = cv2.ORB_create(nfeatures, edgeThreshold=20, patchSize=21)

    # Find keypoints and descriptors.
    kp_ref, descriptors_ref = orb_detector.detectAndCompute(ref, None)
    kp_align, descriptors_align = orb_detector.detectAndCompute(align, None)

    if descriptors_ref is None or descriptors_align is None:
        aligned = align[:, :, 0]
        transformation_matrix = None
    else:
        # Match features between the two images.We create a Brute Force matcher with Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(descriptors_ref, descriptors_align)

        # Sort matches on the basis of their Hamming distance.
    #    matches.sort(key=lambda x: x.distance)
        matches = sorted(matches, key=lambda x: x.distance)

        # Take the top n % matches forward.
        matches = matches[:int(len(matches) * top_matches)]
        no_of_matches = len(matches)

        if view_matches is True:
            imMatches = cv2.drawMatches(ref, kp_ref, align, kp_align, matches, None)
            cv2.imshow("Matched Keypoints", imMatches)
            cv2.waitKey(0)


    # Define empty matrices of shape no_of_matches * 2.
        p_ref = np.zeros((no_of_matches, 2))
        p_align = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p_ref[i, :] = kp_ref[matches[i].queryIdx].pt
            p_align[i, :] = kp_align[matches[i].trainIdx].pt

        #print(p_ref.shape)

        #Proceed only if matches are found. If no matches are found, keep image as it is.
        if no_of_matches == 0:
            aligned = align[:, :, 0]
        else:
            # If found, compute transformation matrix required to go from the aligned image to the reference.
            transformation_matrix, inliers = cv2.estimateAffinePartial2D(p_align, p_ref)
            # Apply transformation matrix and return aligned image
            if transformation_matrix is None:
                aligned = align[:, :, 0]
            else:
                aligned = cv2.warpAffine(toalign, transformation_matrix, (reference.shape[0], reference.shape[1]))
    aligned = aligned.astype(data_type)
    return aligned, transformation_matrix

def ORB_reg_rotation(reference, toalign, nfeatures = 100000, data_type = 'int16', view_matches = False, top_matches = 0.5, return_warped=False):

    '''
    This function uses ORB to register images and correct for the shift between sequencing cycles. Other solutions exist,
    this seemed to be the most straight forward for me. The method does not perform well if not enough key points are selected.
    In my experience this has to be at least 100000.
    :param reference: 2D array -> 2D image that serves as a reference for alignment
    :param toalign: 2D array -> 2D image to be aligned
    :param nfeatures: number of features that ORB uses. Increasing the number helps with the overlap detection
    :param data_type: type of data for final array. not clear to me yet if uint8 is sufficient or should use float 32 or 64.
    :return: 2D array -> 2D image aligned to reference
    '''

    # ORB is finicky and wants 3D arrays so adding another dimension to the input arrays. Also, ORB is double finicky and
    # preferes unsigned ints. I could be wrong.

    ref = np.empty((reference.shape[0], reference.shape[1], 1), np.uint8)
    align = np.empty((toalign.shape[0], toalign.shape[1], 1), np.uint8)
    ref[:, :, 0] = reference
    align[:, :, 0] = toalign

    # Create ORB detector with n features. If problematic, try increasing the number of features.
    orb_detector = cv2.ORB_create(nfeatures, edgeThreshold=20, patchSize=21)

    # Find keypoints and descriptors.
    kp_ref, descriptors_ref = orb_detector.detectAndCompute(ref, None)
    kp_align, descriptors_align = orb_detector.detectAndCompute(align, None)

    if descriptors_ref is None or descriptors_align is None:
        aligned = align[:, :, 0]
        transformation_matrix = None
    else:
        # Match features between the two images.We create a Brute Force matcher with Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(descriptors_ref, descriptors_align)

        # Sort matches on the basis of their Hamming distance.
    #    matches.sort(key=lambda x: x.distance)
        matches = sorted(matches, key=lambda x: x.distance)

        # Take the top n % matches forward.
        matches = matches[:int(len(matches) * top_matches)]
        no_of_matches = len(matches)

        if view_matches is True:
            imMatches = cv2.drawMatches(ref, kp_ref, align, kp_align, matches, None)
            cv2.imshow("Matched Keypoints", imMatches)
            cv2.waitKey(0)


    # Define empty matrices of shape no_of_matches * 2.
        p_ref = np.zeros((no_of_matches, 2))
        p_align = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p_ref[i, :] = kp_ref[matches[i].queryIdx].pt
            p_align[i, :] = kp_align[matches[i].trainIdx].pt

        #print(p_ref.shape)

        #Proceed only if matches are found. If no matches are found, keep image as it is.
        if no_of_matches == 0:
            aligned = align[:, :, 0]
        else:
            # If found, compute transformation matrix required to go from the aligned image to the reference.
            # choose 4 random points in p_ref and the corresponding points in p_align
            rand_idx = np.random.choice(p_ref.shape[0], 3, replace=False)
            p_ref_rand = p_ref[rand_idx, :]
            p_align_rand = p_align[rand_idx, :]
            #transformation_matrix, inliers = cv2.getAffineTransform(p_align_rand, p_ref_rand)

            transformation_matrix, inliers = cv2.estimateAffinePartial2D(p_align, p_ref)
            #transformation_matrix, inliers = cv2.estimateAffine2D(p_align_rand, p_ref_rand)
            #transformation_matrix, inliers = cv2.estimateAffinePartial2D(p_align_rand, p_ref_rand)
            #transformation_matrix = cv2.getPerspectiveTransform(p_align, p_ref)
            #transformation_matrix, _ = cv2.findHomography(p_align, p_ref)
            # Apply transformation matrix and return aligned image
            if transformation_matrix is None:
                aligned = align[:, :, 0]
            else:
                #aligned = cv2.warpPerspective(toalign, transformation_matrix, (reference.shape[0], reference.shape[1]))
                aligned = cv2.warpAffine(toalign, transformation_matrix, (reference.shape[0], reference.shape[1]))
    aligned = aligned.astype(data_type)
    return aligned, transformation_matrix

def PhaseCorr_reg(reference, toalign, return_warped=False):

    reference = reference.astype(np.uint8)
    toalign = toalign.astype(np.uint8)
    image_size = reference.shape

    shifts = skimage.registration.phase_cross_correlation(reference, toalign, reference_mask=~np.isnan(reference),
                                                          moving_mask=~np.isnan(toalign))
    transformation_matrix = np.array([[1, 0, -shifts[1]], [0, 1, -shifts[0]]], dtype=np.float)
    if return_warped is True:
        aligned = cv2.warpAffine(toalign, transformation_matrix, (image_size[1], image_size[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned, transformation_matrix
    else:
        return 0, transformation_matrix

def ECC_reg(reference, toalign, number_of_iterations = 2000, termination_eps = 1e-10, return_warped=False):
    '''
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    :param reference:
    :param toalign:
    :param number_of_iterations:
    :param termination_eps:
    :return:
    '''
    reference = np.uint8(reference)
    toalign = np.uint8(toalign)

    image_size = reference.shape
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (_, warp_matrix) = cv2.findTransformECC(reference, toalign, warp_matrix, warp_mode, criteria, None, 5)
    except cv2.error:
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float)
        print('did not converge')

    if return_warped is True:
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(toalign, warp_matrix, (image_size[1], image_size[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(toalign, warp_matrix, (image_size[1], image_size[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned, warp_matrix
    else:
        return 0, warp_matrix

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def list_files(path):
    files = os.listdir(path)
    if '.DS_Store' in files: files.remove('.DS_Store')
    return files

def quick_dir(location, folder_name):
    '''
    Check if a directory exists, otherwise creates one
    :param location:
    :param name:
    :return:
    '''

    if location[-1] == '/': folder_path = location + folder_name + '/'
    else: folder_path = location + '/' + folder_name + '/'

    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)

    return folder_path

def saveAsTXT(coord, filename, destination):

    txtfilename = destination + '/' + filename + '.txt'
    coord = coord.astype(int)
    np.savetxt(txtfilename, coord, '%5.0f', delimiter='\t')

def find_masks_2D(filepath):

    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    color_mask = tif.imread(filepath)
    color_mask = color_mask.astype(np.uint8)
    ret, thresh = cv2.threshold(color_mask, 1, 255, 0)

    lbl, nr_blobs = scp.ndimage.label(thresh)
    CoM = scp.ndimage.measurements.center_of_mass(thresh, lbl, np.arange(nr_blobs))
    CoM = np.array(CoM[1:])
    #CoM = pad_n_cols_right_of_2d_matrix(CoM, 1)

    return CoM

def find_stack_rotation(slice_points, stack_points):
    '''
    Function that finds the rotation between two vectors. designed to be used to find the rotation between slices and the in vivo stack.
    Input should be 3x3 numpy arrays representing 3 points with xyz coordinated in stack and slice
    :param slice_points: 3x3 numpy arrays with xyz coords of 3 distinctive points in slice
    :param stack_points: 3x3 numpy arrays with xyz coords of the same 3 points in vivo
    :return: rotation in degrees with respect to x, y, and z axis.
    '''

    assert slice_points.shape == (3, 3), f"A 3x3 matrix is expected for slice_points"
    assert stack_points.shape == (3, 3), f"A 3x3 matrix is expected for stack_points"

    # Create 2 vectors in plane out of the 3 points
    slice_vec1 = slice_points[2] - slice_points[1]
    slice_vec2 = slice_points[1] - slice_points[0]

    stack_vec1 = stack_points[2] - stack_points[1]
    stack_vec2 = stack_points[1] - stack_points[0]

    # find cross product of the 2 vectors we found
    slice_cross = np.cross(slice_vec2, slice_vec1)
    stack_cross = np.cross(stack_vec2, stack_vec1)
    #    print('difference in vectors is', slice_cross-stack_cross)

    # normalize cross product. this step is optional but help with debugging if results look weird.
    slice_cross = slice_cross / np.linalg.norm(slice_cross)
    stack_cross = stack_cross / np.linalg.norm(stack_cross)

    slice_cross = slice_cross.reshape((1, 3))
    stack_cross = stack_cross.reshape((1, 3))

    # find rotation between the two cross products
    # rotation_matrix = scipy.spatial.transform.Rotation.align_vectors(slice_cross, stack_cross)
    # print(rotation_matrix[0].as_matrix())
    rotation_matrix = rotation_matrix_from_vectors(slice_cross, stack_cross)
    rotation_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
    # express rotation in degrees
    # x_rot, y_rot, z_rot = rotation_matrix[0].as_euler('XYZ', degrees=True)
    x_rot, y_rot, z_rot = rotation_matrix.as_euler('XYZ', degrees=True)
    return x_rot, y_rot, z_rot


def find_cellpose_contours(mask):

    nr_labels = np.max(np.max(mask))
    contours = []
    centers_of_mass = []
    for thresh in range(1, nr_labels + 1):
        thresh_mask = np.ones_like(mask, dtype=np.uint8) * 10
        thresh_mask = thresh_mask * (mask == thresh)
        ret, thresh = cv2.threshold(thresh_mask, 1, 255, 0)
        contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contour[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        coords = np.vstack(contour[0])

        contours.append(coords)
        centers_of_mass.append([cX, cY])

    contours = np.array(contours)

    return contours, centers_of_mass

def find_cellpose_contours_overlapping_masks(mask_nuclei, mask_cyto):

    max_nuclei = np.max(np.max(mask_nuclei))
    max_cyto = np.max(np.max(mask_cyto))

    #increment all values greater than 0 in cyto mask by the max value in nuclei mask
    greater_than_0 = mask_cyto > 0
    mask_cyto[greater_than_0] = mask_cyto[greater_than_0] + max_nuclei

    mask = np.max(np.stack((mask_nuclei, mask_cyto)), axis=0)
    #nr labels is the number of unique values in the mask
    unique_vals = np.unique(mask)
    #exclude 0 if present
    if 0 in unique_vals: unique_vals = unique_vals[1:]
    nr_labels = len(unique_vals)
    contours = []
    centers_of_mass = []
    for thresh in unique_vals:
        thresh_mask = np.ones_like(mask, dtype=np.uint8) * 10
        thresh_mask = thresh_mask * (mask == thresh)
        ret, thresh = cv2.threshold(thresh_mask, 1, 255, 0)
        contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contour[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        coords = np.vstack(contour[0])

        contours.append(coords)
        centers_of_mass.append([cX, cY])

    contours = np.array(contours)

    return contours, centers_of_mass, mask


def find_contours(stack):
    if stack.ndim == 2:
        no_channels = 1
    else:
        no_channels = stack.shape[0]
#    channels = [stack[channel] for channel in range(no_channels)]
    stack = stack.astype(np.uint8)
    stack_contours = []
    stack_centers_of_mass = []
    for id in range(no_channels):
        ret, thresh = cv2.threshold(stack[id], 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers_of_mass = np.ndarray((len(contours), 2))
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            centers_of_mass[i] = [cX, cY]
        stack_contours.append(contours)
        stack_centers_of_mass.append(centers_of_mass)

    return stack_contours, stack_centers_of_mass

def find_contours_3D(stack):
    stack = stack.astype(np.uint8)
    print('stack shape ', stack.shape)
    if stack.ndim == 3:
        stack = np.transpose(stack, [1,2,0])
    ret, thresh = cv2.threshold(stack, 1, 256, 0)
    lbl, nr_blobs = scp.ndimage.label(thresh)
    print('label shape', lbl.shape)
    CoM = scp.ndimage.measurements.center_of_mass(thresh, lbl, np.arange(nr_blobs))
    CoM = np.array(CoM)
    print('stack has ', CoM.shape, ' points')
    return CoM

def find_overlapping_contours(target, source, allowed_shift=10):
    dist = cdist(target, source, metric='euclidean')
    dist = np.delete(dist, 0, axis=0)
    dist = np.delete(dist, 0, axis=1)
    indices = np.argmin(dist, axis=1)
    minvals = np.min(dist, axis=1)

    minvals = minvals.reshape(len(indices), 1)
    indices = indices.reshape(len(indices), 1)
    add_column = np.arange(0, indices.shape[0])
    add_column = add_column.reshape(indices.shape[0], 1)

    overlapping_CoM = np.column_stack((add_column, indices, minvals))
    overlapping_CoM = overlapping_CoM[overlapping_CoM[:, 2] < allowed_shift]

    return overlapping_CoM

def find_stack_rotation(slice_points, stack_points):
    '''
    Function that finds the rotation between two vectors. designed to be used to find the rotation between slices and the in vivo stack.
    Input should be 3x3 numpy arrays representing 3 points with xyz coordinated in stack and slice
    :param slice_points: 3x3 numpy arrays with xyz coords of 3 distinctive points in slice
    :param stack_points: 3x3 numpy arrays with xyz coords of the same 3 points in vivo
    :return: rotation in degrees with respect to x, y, and z axis.
    '''

    assert slice_points.shape == (3, 3), f"A 3x3 matrix is expected for slice_points"
    assert stack_points.shape == (3, 3), f"A 3x3 matrix is expected for stack_points"

    # Create 2 vectors in plane out of the 3 points
    slice_vec1 = slice_points[2] - slice_points[1]
    slice_vec2 = slice_points[1] - slice_points[0]

    stack_vec1 = stack_points[2] - stack_points[1]
    stack_vec2 = stack_points[1] - stack_points[0]

    # find cross product of the 2 vectors we found
    slice_cross = np.cross(slice_vec2, slice_vec1)
    stack_cross = np.cross(stack_vec2, stack_vec1)
    #    print('difference in vectors is', slice_cross-stack_cross)

    # normalize cross product. this step is optional but help with debugging if results look weird.
    slice_cross = slice_cross / np.linalg.norm(slice_cross)
    stack_cross = stack_cross / np.linalg.norm(stack_cross)

    slice_cross = slice_cross.reshape((1, 3))
    stack_cross = stack_cross.reshape((1, 3))

    # find rotation between the two cross products
    # rotation_matrix = scipy.spatial.transform.Rotation.align_vectors(slice_cross, stack_cross)
    # print(rotation_matrix[0].as_matrix())
    rotation_matrix = rotation_matrix_from_vectors(slice_cross, stack_cross)
    rotation_matrix = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
    # express rotation in degrees
    # x_rot, y_rot, z_rot = rotation_matrix[0].as_euler('XYZ', degrees=True)
    x_rot, y_rot, z_rot = rotation_matrix.as_euler('XYZ', degrees=True)
    return x_rot, y_rot, z_rot


def create_illumination_profiles(folder_path, nr_images=200):
    '''
    This functions take as input a folder with nr_images and computes flatfield and darkfield profiles.
    :param folder:
    :param nr_images:
    :return: flatfield, darkfield
    '''

    import pybasic
    tiles = helpers.list_files(folder_path)
    #choose last nr_images images, or all images if nr_images is larger than the number of images in the folder
    nr_images = min(nr_images, len(tiles))
    tiles = tiles[-nr_images:]
    sample_image = tif.imread(folder_path + tiles[0])
    nr_channels = sample_image.shape[0]
    flatfield = np.zeros((nr_channels, sample_image.shape[1], sample_image.shape[2]))
    darkfield = np.zeros((nr_channels, sample_image.shape[1], sample_image.shape[2]))

    tiles_stack = np.zeros((nr_channels, nr_images, sample_image.shape[1], sample_image.shape[2]))

    for i, tile in enumerate(tiles):
        tiles_stack[:, i] = tif.imread(folder_path + tile)

    for i in range(nr_channels):
        flatfield[i], darkfield[i] = pybasic.basic(tiles_stack[i], darkfield=True)
        #print ranges of flatfield and darkfield
        #flatfield[i] = np.mean(tiles_stack[i], axis=0)
        # normalize flatfield to 1
        #flatfield[i] = flatfield[i] / np.max(flatfield[i])
    return flatfield, darkfield


def remove_outstringers(list, substring):

    list_to_remove = []
    for string_name in list:
        if substring not in string_name:
            list_to_remove.append(string_name)
    for item in list_to_remove:
        list.remove(item)
    del list_to_remove
    return list

def get_trailing_number(s):
    '''
    Returns the number at the end of a string. Useful when needing to extract the sequencing cycle number from folder name.
    Input:
        s (string): name containing number
    Returns:
        integer at the end of string
    '''
    import re
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def sort_position_folders(positions_list):

    segment = re.search('(.*)Pos(.*)', positions_list[0])
    starting_segment = segment.group(1)

    positions_dict = {}
    for position_name in positions_list:
        segment = re.search(starting_segment + 'Pos(.*)_(.*)_(.*)', position_name)
        pos = 'Pos' + segment.group(1);
        if pos in positions_dict:
            positions_dict[pos].append(position_name)
        else:
            positions_dict[pos] = [position_name]

    return positions_dict

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def human_sort(list):
    list.sort(key=natural_keys)
    return list
def check_images_overlap(imageA, imageB, save_output=True, path=None, folder_name='overlap_check', filename='images_overlap.tif'):

    '''
    Function that take 2 images and combines them into a 2 chan stack to visualize alignment. Voth get converted to int16
     and then histograms are equalized to help with different intensity ranges.
    :param imageA: first image
    :param imageB: second image
    :param save_output: do you want to save input to a folder of choice?
    :param path: if so, specify folder
    :param folder_name: if doing it for multiple pairs, you may want to give folder a different name. otherwise 'overlap_check' it is
    :return: returns the 2 chan stack
    '''

    axis_norm = (0, 1)
    imageA = normalize(imageA, 1, 99.8, axis=axis_norm)
    imageB = normalize(imageB, 1, 99.8, axis=axis_norm)
    images_overlap = np.stack((imageA, imageB), axis=0)

    if save_output is True:
        overlap_check_path = quick_dir(path, folder_name)
        tif.imsave(overlap_check_path + filename, images_overlap)
    return images_overlap

def check_images_overlap_trials(imageA, imageB, save_output=True, path=None, folder_name='overlap_check', filename='images_overlap.tif'):

    '''
    Function that take 2 images and combines them into a 2 chan stack to visualize alignment. Voth get converted to int16
     and then histograms are equalized to help with different intensity ranges.
    :param imageA: first image
    :param imageB: second image
    :param save_output: do you want to save input to a folder of choice?
    :param path: if so, specify folder
    :param folder_name: if doing it for multiple pairs, you may want to give folder a different name. otherwise 'overlap_check' it is
    :return: returns the 2 chan stack
    '''

    #if imageA.dtype != 'int16': imageA = imageA.astype(np.int16)
    #if imageB.dtype != 'int16': imageB = imageB.astype(np.int16)
    #imageA = (imageA / 256).astype('uint8')
    #imageB = (imageB / 256).astype('uint8')

    #maxA = np.max(np.max(imageA))
    #maxB = np.max(np.max(imageB))
    #imageA = imageA * (maxB/maxA)
    #imageA = skimage.exposure.equalize_hist(imageA)
    #imageB = skimage.exposure.equalize_hist(imageB)
    axis_norm = (0, 1)
    imageA = normalize(imageA, 1, 99.8, axis=axis_norm)
    imageB = normalize(imageB, 1, 99.8, axis=axis_norm)
    #imageA = cv2.equalizeHist(imageA)
    #imageB = cv2.equalizeHist(imageB)
    #imageA = (65535 * imageA).round().astype(np.uint16)
    #imageB = (65535 * imageB).round().astype(np.uint16)

    #if imageA.dtype != 'int16': imageA = imageA.astype(np.int16)
    #if imageB.dtype != 'int16': imageB = imageB.astype(np.int16)

    images_overlap = np.stack((imageA, imageB), axis=0)
    #images_overlap = images_overlap.astype(np.int16)

    #RGB_image = np.stack((imageA, imageB, imageB), axis=0)

    if save_output is True:
        overlap_check_path = quick_dir(path, folder_name)
        tif.imsave(overlap_check_path + filename, images_overlap)
        #tif.imsave(overlap_check_path + 'RGB_overlap.tif', RGB_image)
    return images_overlap

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def crop_center_image(img, cropx, cropy):
    '''
    Crops image with respect to center coordinates
    :param img:
    :param cropx:
    :param cropy:
    :return:
    '''
    if img.ndim == 3:
        if img.shape[0] < img.shape[2]:
            _, y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[:, starty:starty + cropy, startx:startx + cropx]
        else:
            y, x, _ = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx, :]
    elif img.ndim == 2:
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

def matrix_of_None(rows, cols):
    matrix = []
    for i in range(rows):
        row_tiles = []
        for j in range(cols):
            row_tiles.append(None)
        matrix.append(row_tiles)
    return matrix

def pad_n_cols_left_of_2d_matrix(arr, n):
    """Adds n columns of zeros to left of 2D numpy array matrix.

    :param arr: A two dimensional numpy array that is padded.
    :param n: the number of columns that are added to the left of the matrix.
    """
    padded_array = np.zeros((arr.shape[0], arr.shape[1] + n))
    padded_array[:, n:] = arr
    return padded_array


def pad_n_cols_right_of_2d_matrix(arr, n):
    """Adds n columns of zeros to right of 2D numpy array matrix.

    :param arr: A two dimensional numpy array that is padded.
    :param n: the number of columns that are added to the right of the matrix.
    """
    padded_array = np.zeros((arr.shape[0], arr.shape[1] + n))
    padded_array[:, : arr.shape[1]] = arr
    return padded_array


def pad_n_rows_above_2d_matrix(arr, n):
    """Adds n rows of zeros above 2D numpy array matrix.

    :param arr: A two dimensional numpy array that is padded.
    :param n: the number of rows that are added above the matrix.
    """
    padded_array = np.zeros((arr.shape[0] + n, arr.shape[1]))
    padded_array[n:, :] = arr
    return padded_array


def pad_n_rows_below_2d_matrix(arr, n):
    """Adds n rows of zeros below 2D numpy array matrix.

    :param arr: A two dimensional numpy array that is padded.
    :param n: the number of rows that are added below the matrix.
    """
    padded_array = np.zeros((arr.shape[0] + n, arr.shape[1]))
    padded_array[: arr.shape[0], :] = arr
    return padded_array

def append_value_to_file(path, filename, value, txt_format='%i'):
    file_array = np.load(path + filename + '.npy')
    rows, cols = file_array.shape
    assert value.size == cols, f'cannot concatenate value of shape ' + str(value.size) + 'to array of shape' + file_array.shape
    file_array = np.concatenate(file_array, value, axis=1)
    np.savetxt(path + filename + '.txt', file_array, fmt=txt_format)
    np.save(path + filename + '.npy', file_array)



# Rotates 3D image around image center
# INPUTS
#   array: 3D numpy array
#   orient: list of Euler angles (phi,psi,the)
# OUTPUT
#   arrayR: rotated 3D numpy array
# by E. Moebel, 2020
def rotate_array(array, orient):
    rot = orient[0]
    tilt = orient[1]
    phi = orient[2]

    # create meshgrid
    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    r = R.from_euler('ZXY', [rot, tilt, phi], degrees=True)
    mat = r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape((dim[1], dim[0], dim[2]))  # I test the rotation in 2D and this strange thing can be explained
    new_xyz = [x, y, z]

    arrayR = map_coordinates(array, new_xyz, order=1).T
    return arrayR

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions


def get_3Drotation(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def scale_to_8bit(img, unsigned=True):
    img = img.astype(np.float32)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    if unsigned:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.int8)
    return img

def create_matcher_profile():
    # creates a pop-up box where user can input their name
    root = tk.Tk()
    root.title('Matcher name')
    canvas = tk.Canvas(root, width=300, height=300)
    canvas.pack()

    entry_box = tk.Entry(root)
    # display text 'Input your name' in the pop-up box
    canvas.create_window(150, 150, window=entry_box)
    # display message
    tk.Label(root, text='So, you want to be a matcher? Tell me your name and it is done.').pack()
    # create a button to close the pop-up box

    def get_name():
        global name
        name = entry_box.get()
        # display a message in response
        tk.Label(root, text='Welcome to the team, ' + name + '!').pack()
        root.destroy()

    button = tk.Button(text='Submit my name', command=get_name)
    canvas.create_window(150, 180, window=button)
    root.mainloop()
    return name

def back_up_folders(destination, list_of_folders, list_of_files):
    #get time stamp
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #create a folder with the time stamp
    time_stamp_path = helpers.quick_dir(destination, time_stamp)

    for folder in list_of_folders:
        #copy the folder to the time stamp folder
        shutil.copytree(folder, time_stamp_path + folder.split('/')[-2])
    for file in list_of_files:
        # if file exists
        if os.path.isfile(file):
        #copy the file to the time stamp folder
           shutil.copyfile(file, time_stamp_path + file.split('/')[-1])
