# Author: Nick Sebasco
# Date: (4/17-4/21)/2021
# Use kmeans clustering to segement lip image.
# end goal will be to autosample lip and then
# classify as either lesion or no lesion.
# v5 - takes all samples, code cleaned
# [x] calculate accuracy from v4:
# 49/51 = 96%
# [x] Implement dynamic bounding box size.
#   the size of the bounding rect now grows with the image height.
# [x] choose best sample algorithm -> take sample above, below, left, right.
#   the sample with the best color match is the sample we want to save.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir, path as ospath, remove as force_remove_file
import os
from random import choice
from PIL.Image import fromarray 
#---------------------------------------------
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave

# 0. globals
path = "Oral Lesion Database"
dir = ospath.join("..","..",path, "Normal_Lips")
gamma = None
rows, cols = (2,2)
AVG_LIP_COLOR = [204, 130, 124] # calculated via lip_color.py

# 1. helper functions
def average_rgb(mat, median=False):
    m,n,k = mat.shape
    r,g,b = [],[],[]
    for i in range(m):
        for j in range(n):
            r.append(float(mat[i,j][2]))
            g.append(float(mat[i,j][1]))
            b.append(float(mat[i,j][0]))

    return [np.median(l) if median else np.mean(l) for l in [r,g,b]]

def dist(x, y):
    """euclidean distance
    used by dist_pixel to find the closest pixel to a reference pixel."""
    t = 0
    for i,j in zip(x,y):
        t += (i - j)**2
    return t**0.5

def dist_pixel(mat, test = [0, 0, 0]):
    """Find pixel in matrix with minimum distance to test.
    This is a way of finding the pixel in an image that has 
    the most similar color as test."""
    s = set()
    for x in mat:
        for y in x:
            s.add(tuple(y))
    return min(s,key=lambda x: dist(x, test))

def darkest_pixel(mat):
    """Find darkest pixel in image: min(sum([r,g,b])).
    """
    s = set()
    for x in mat:
        for y in x:
            s.add(tuple(y))
    return min(s,key=lambda x: sum(x))

def filter_img(mat, criteria, default = [255, 255, 255]):
    """Force all pixels not equal to criteria to be white.
    """
    m,n,k = mat.shape
    for i in range(m):
        for j in range(n):
            mat[i,j] = mat[i,j] if tuple(mat[i,j]) == criteria else default

def dist_from_center(mat):
    """Find the distance & pixel location, of the darkest pixel
    from the center of the image.
    """
    m,n,k = mat.shape
    center = (m//2,n//2)
    print("center: ",center)
    min_dist, min_locale = float("inf"), ()
    dp = list(darkest_pixel(mat))
    for i in range(m):
        for j in range(n):
            if list(mat[i,j]) ==  dp:
                di = dist(center, (i,j))
                if di < min_dist:
                    min_dist = di
                    min_locale = (i, j)
    return (min_dist, min_locale)

def gamma_correction(image, gamma):
    """build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table) 

def increase_brightness(img, value=30):
    """Function to increase the brightness of an image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def smooth_image(img, kernel_size = (5, 5), blur_type="blur"):
    """Apply blurring techniques in an effort to enhance structures in image.
    blur_type: blur, median, gaussian"""
    if blur_type == "blur":
        return cv2.blur(img, kernel_size)
    elif blur_type == "median":
        return cv2.medianBlur(img,kernel_size[0])
    elif blur_type == "guassian":
       return cv2.GaussianBlur(img,kernel_size,0) 

def apply_sk_threshold(dir, test, name='test.png'):
    """
    Use SK learn to apply thresholding.  SKlearn creates new temp file
    which is read by cv2, then deleted.
    """
    img = imread(ospath.join(dir, test))
    image_backup = img.copy()
    yen_threshold = threshold_yen(img)
    bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
    imsave(name, bright)
    # Read in the image
    image = cv2.imread(name)
    force_remove_file(name)
    return (image, image_backup)

def read_image(dir, test, name='test.png',apply_thresholding=True):
    """
    We can either apply thresholding to our image after reading or not.
    """
    if apply_thresholding:
        return apply_sk_threshold(dir, test, name=name)
    else:
        img = imread(ospath.join(dir, test))
        return (img, img.copy())

def create_sample(img, coords, fpath, resize = (50,50), bound_rect_img = None):
    """ Create sample of labial vermillion tissue.
    """
    cropped_image = fromarray(img).crop(coords).resize(resize,0)
    cropped_image.save(fpath)
    if type(bound_rect_img) != type(None):
        bound_rect_img_obj = fromarray(bound_rect_img())
        bound_rect_img_obj.save(fpath.replace(".png","_bound_rect.png"))

def find_best_sample(img, min_locale, start_off, off):
    spoints = [
        (min_locale[0] - start_off,min_locale[1] - start_off)[::-1],
        (min_locale[0] + start_off,min_locale[1] + start_off)[::-1],
        (min_locale[0] + start_off,min_locale[1] - start_off)[::-1],
        (min_locale[0] - start_off,min_locale[1] + start_off)[::-1]
    ]
    epoints = [
        (spoints[0][0] - off, spoints[0][1] - off),
        (spoints[1][0] + off, spoints[1][1] + off),
        (spoints[2][0] + off, spoints[2][1] - off),
        (spoints[3][0] - off, spoints[3][1] + off)
    ]
    res = []
    i = 0
    for start_point, end_point in zip(spoints, epoints):
        #print("start: ", start_point, "end: ", end_point)
        if i == 0:
            cropped_image = fromarray(img.copy()).crop((*start_point, *end_point))
        elif i == 1:
            cropped_image = fromarray(img.copy()).crop((*end_point, *start_point))
        elif i == 2:
            coords = (min((start_point[0], end_point[0])),min((start_point[1], end_point[1])),max((start_point[0], end_point[0])),max((start_point[1], end_point[1])))
            cropped_image = fromarray(img.copy()).crop(coords)
        elif i == 3:
            coords = (min((start_point[0], end_point[0])),min((start_point[1], end_point[1])),max((start_point[0], end_point[0])),max((start_point[1], end_point[1])))
            cropped_image = fromarray(img.copy()).crop(coords)
        cropped_image = cropped_image.convert("RGB")
        cropped_image = np.asarray(cropped_image, dtype=np.float32)
        cropped_image = cropped_image[:, :, :3]
        v_rgb = average_rgb(cropped_image)
        dv = dist(v_rgb, AVG_LIP_COLOR)
        #print("avg_rgb: ",v_rgb)
        #print("dist: ", dv)
        res.append([dv, (start_point, end_point)])
        i += 1
    return min(res,key=lambda x: x[0])[1] # grab only the start/ end points

def find_optimal_kval(img):
    import seaborn as sns
    from sklearn.cluster import KMeans 
    import matplotlib.pyplot as plt
    cluster_mss = []
    k_values = range(1, 20)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(img)
        cluster_mss.append(kmeans.inertia_)
    sns.set()
    sns.lineplot(k_values, cluster_mss, markers=True,  marker="o", color="red")
    plt.axvline(x=3, linestyle= ':')
    # plt.plot(k_values, cluster_mss, "ro")
    # plt.plot(k_values, cluster_mss, "r")
    plt.title("Elbow curve")
    plt.ylabel("MSS for each point wrt center")
    plt.xlabel("Value of K")
    plt.show()

# 3. main function
def main(
    axarr, # allows for generating subplots
    i = 5,
    k = 2, # the k parameter for the k-means algorithm
    kernel_size= (3,3), # kernel size used for smoothing algorithms
    coords = (0,0),
    gamma=None,
    apply_thresholding=False,
    add_brightness=False,
    showMasked=False,
    sample_opts = {
        "collect": False,
        "save_bounding_rect": True
        },
    showPlots=True,
    dynamicOffset=True,
    showBoundingRect=False,
    find_optimal_k=False
    ):
    """The main logic for the program.
    """
    test = choice(listdir(dir)) if False else listdir(dir)[i]
    image, image_backup = read_image(dir, test)

    # increase brightness (optional)
    if add_brightness:
        image = increase_brightness(image,apply_thresholding=True)

    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if find_optimal_k:
        x, y, z = image.shape
        img = image.reshape(x*y, z)
        find_optimal_kval(img)
        return 

    # smooth the image to inhance structures
    image = smooth_image(image, kernel_size=kernel_size,blur_type="median")

    # apply gamma correction
    if gamma:
        image = gamma_correction(image, gamma)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1,3))
    
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    print(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    
    # then perform k-means clustering wit h number of clusters defined as 3
    #also random centres are initally chosed for k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    #print(darkest_pixel(segmented_image))
    filter_img(segmented_image,dist_pixel(segmented_image, [0,0,0,0]))

    if showMasked: 
        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(image)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        # color (i.e cluster) to disable
        cluster = 2
        masked_image[labels.flatten() == cluster] = [0, 0, 0]
        # convert back to original shape
        masked_image = masked_image.reshape(image.shape)
        plt.imshow(masked_image)
        plt.show()

    min_dist, min_locale = dist_from_center(segmented_image)
    print(f"min distance: {min_dist} @ {min_locale}")

    if showBoundingRect:
        # Start coordinate, here (100, 50)
        # represents the top left corner of rectangle
        start_off = -5
        off = -10
        if dynamicOffset:
            m,n,k = image.shape
            off = off + (m//100 * off)
            start_off = start_off + (m//100 * start_off)

        start_point0 = (min_locale[0] - start_off,min_locale[1] - start_off)[::-1]
        print(image.shape)
        #print("start: ", start_point0)
        # Ending coordinate, here (125, 80)
        # represents the bottom right corner of rectangle
        end_point0 = (start_point0[0] - off, start_point0[1] - off)#(start_point[0] + off, start_point[1] + off)[::-1]
        #print("end:", end_point0)
        # Black color in BGR
        color = (0, 0, 0)
        
        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = 1

        #print("Call from sample: ")
        start_point, end_point = find_best_sample(image_backup.copy(), min_locale, start_off, off)
        #print("FOUND best sample: ", (start_point, end_point))
        #print("points =", start_point0 == start_point)

        # create sample:
        if sample_opts["collect"]:
            if sample_opts["save_bounding_rect"]:
                # create new directory to store image + cropping
                if not os.path.exists(f"samples/normalLip_sample_{i}"): os.mkdir(f"samples/normalLip_sample_{i}")
                # create lip tissue sample & bounding rect image
                # pass bound_rect_img as a function so it does not mutate the sample I want to collect.
                create_sample(image_backup, (*start_point,*end_point),
                f"samples/normalLip_sample_{i}/normalLip_sample_{i}.png",
                bound_rect_img=lambda:cv2.rectangle(image_backup, start_point, end_point, color, thickness))
            else:
                create_sample(image_backup, (*start_point,*end_point),f"samples/normalLip_sample_{i}.png")
                
        # Using cv2.rectangle() method
        # Draw a rectangle of black color of thickness -1 px
        image = cv2.rectangle(image_backup, start_point, end_point, color, thickness)
        plt.imshow(image)
        if showPlots:
            plt.show()

    #print("labels:", labels, len(labels))
    #print(segmented_image)
    if type(axarr) != type(None):
        #axarr[coords[0], coords[1]].set_title(f"kernel size: {kernel_size[0]}")
        #axarr[coords[0], coords[1]].imshow(segmented_image)
        plt.imshow(image)
        plt.show()

# 3.1.  use main to search for optimal hyperparameters.
def test():
    """test for optimal hyperparameter values: kernel_size, k, gamma, smoothing algorithm, thresholding, etc.
    """
    for j in range(2,3):
        m,n = (0,0)
        f, axarr = plt.subplots(2,2)
        for i in range(1,5):
            print(m,n)
            main(axarr, i=1, k=j, kernel_size=(2*i+1,2*i+1), coords=(m,n), gamma=gamma,find_optimal_k=True)
            if i % cols == 0:
                n = 0
                m += 1
            else:
                n += 1

# 3.2. use main to collect samples over an entire directory         
def collect_sample():
    """Procedure used to initiate sample collection.
    """
    N = len(listdir(dir))
    for j in range(N):
        main(axarr=None, 
        i=j, 
        k=3, 
        kernel_size=(3, 3),
        gamma=gamma,
        apply_thresholding=True,
        showMasked=False,
        sample_opts = {
        "collect": True,
        "save_bounding_rect": True
        },
        showPlots=True,
        showBoundingRect=True)
        #plt.show()

if True:
    test()
