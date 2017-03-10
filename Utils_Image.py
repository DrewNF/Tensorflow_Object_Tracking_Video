

from PIL import Image, ImageChops,ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import cv2
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import webcolors


size = (640,480)
img_save_type='PNG'

### Function to check the existence of an image

def check_image_with_pil(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True

def get_Image_List(path, ext):
    files_list=[]
    for path, subdirs,files in os.walk(path):
        for filename in files:
            if not filename.endswith(ext): continue
            files_list.append(os.path.join(path, filename))
    return files_list

def get_Image_Size(path):

    if check_image_with_pil(path):
        im = Image.open(path)
        width, height = im.size
        return width, height
    else : print 'ERROR: Image not exists in filesystem:%s'%path

def change_extension(file_path, ext_1, ext_2):
    #Resize Cropping & Padding an image to the 640x480 pixel size
    if check_image_with_pil(file_path):
        image = Image.open(file_path)
    #print'Starting Path: %s'% file_path
    
    new_path=file_path.replace(ext_1,ext_2)
        #print'New Path: %s'%  new_path
    image.save(new_path, img_save_type)
    if check_image_with_pil(new_path):
        #print 'Rename & Save completed Correct for: %s'%new_path 
        os.remove(file_path)
    else : print 'ERROR: Rename & Save for: %s'%new_path



### Functions to resize /and save an image

def resizeImage(file_path): 
    #Resize Cropping & Padding an image to the 640x480 pixel size
    if file_path is not -1:
        if check_image_with_pil(file_path):
            image = Image.open(file_path)
            image.thumbnail(size, Image.ANTIALIAS)
            image_size = image.size

            padding_0 = max( (size[0] - image_size[0]) / 2, 0 )
            padding_1 = max( (size[1] - image_size[1]) / 2, 0 )
            cv2.namedWindow('Original Image')
            cv2.namedWindow('Resized Image')
            cv2.startWindowThread()
            orig_img = cv2.imread(file_path, 0)
            cv2.imshow('Original Image',orig_img)
            cv2.waitKey(2)

            if((padding_0==0) & (padding_1==0)):
                image.save(file_path, img_save_type)
            else:
                thumb = image.crop( (0, 0, size[0], size[1]) )
                thumb = ImageChops.offset(thumb, int(padding_0), int(padding_1))
                thumb.save(file_path)

            resized_img = cv2.imread(file_path, 0)
            cv2.imshow('Resized Image',resized_img)
    else :
        cv2.destroyAllWindows()
        cv2.waitKey(2)


def resize_saveImage(file_path, new_path): 
    #Resize Cropping & Padding an image to the 640x480 pixel size
    ##The method thumbnail mantain the aspect ratio and resize the image to fit the max size passed
    ##depending on the orientation of the image.
    ##Than with Image chops we set the smaller ones in 
    
    image = Image.open(file_path)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    padding_0 = max( (size[0] - image_size[0]) / 2, 0 )
    padding_1 = max( (size[1] - image_size[1]) / 2, 0 )

    orig_img = cv2.imread(file_path, 0)
    cv2.imshow('Original Image',orig_img)
    cv2.waitKey(2)

    if((padding_0==0) & (padding_1==0)):
        image.save(new_path, img_save_type)
    else:
        thumb = image.crop( (0, 0, size[0], size[1]) )
        thumb = ImageChops.offset(thumb, int(padding_0), int(padding_1))
        thumb.save(new_path)

    resized_img = cv2.imread(new_path, 0)
    cv2.imshow('Resized Image',resized_img)
    cv2.waitKey(2)

    if not check_image_with_pil(new_path):
        print 'ERROR: Rename & Save for: %s'%new_path
    if check_image_with_pil(file_path):
        os.remove(file_path)
        #print 'Delected old File: %s'%file_path

    return padding


### Function to get the image padd

def getpadd_Image(size_img_0, size_img_1, max_size_0, max_size_1): 
    #Get Padd of the image
    orig_ratio=float(size_img_0/size_img_1)
    new_ratio=-1
    max_ratio=float(max(float(size_img_0/max_size_0),float(size_img_1/max_size_1)))
    new_img_0=int(size_img_0/max_ratio)
    new_img_1=int(size_img_1/max_ratio)
    new_ratio=int(new_img_0/new_img_1)
    if new_ratio is not int(max_ratio):
        print "Ratio Error"
    padding[0] = max( (max_size_0 - new_img_0) / 2, 0 )
    padding[1] = max( (max_size_1 - new_img_1) / 2, 0 )

    return padding

### Functions to manage the point resizing

def transform_point(size_img_0, size_img_1, max_size_0, max_size_1, point, xory):
    orig_ratio=float(size_img_0)/float(size_img_1)
    new_ratio=-1
    max_ratio=float(max(float(size_img_0/max_size_0),float(size_img_1/max_size_1),1))
    # print 'Size W Img: %d'% size_img_0
    # print 'Size H Img: %d'% size_img_1
    # print 'Size MW Img: %d'% max_size_0
    # print 'Size MH Img: %d'% max_size_1
    # print 'Starting Point Img: %d'% point
    # print 'Max Ratio New Img: %d'%max_ratio
    if(max_ratio==1):
        if xory:
            #print "x point"
           padding = max( (max_size_0 - size_img_0) / 2, 0 )
        else:
            #print "y point"    
            padding = max( (max_size_1 - size_img_1) / 2, 0 )
        point = point + padding
    else:   
        new_img_0=int(size_img_0/max_ratio)
        new_img_1=int(size_img_1/max_ratio)
        new_ratio=int(new_img_0/new_img_1)
        old_ratio=int(size_img_0/size_img_1)
        if new_ratio is not old_ratio:
            print "Ratio Error %d : %d"%(new_ratio,old_ratio)
            if xory:
               # print "x point"
               padding = max( (max_size_0 - new_img_0) / 2, 0 )
            else:
               # print "y point"
                padding = max( (max_size_1 - new_img_1) / 2, 0 )
            point = int(point/max_ratio) + padding
    # print 'Padding Point Img: %d'%padding 
    # print 'Ending Point Img: %d'%point
    return point

# def get_orig_point(size_img_0, size_img_1, max_size_0, max_size_1, point, xory):
#     orig_ratio=float(size_img_0/size_img_1)
#     new_ratio=-1
#     max_ratio=float(max(float(size_img_0/max_size_0),float(size_img_1/max_size_1),1))
#     # print 'Size W Img: %d'% size_img_0
#     # print 'Size H Img: %d'% size_img_1
#     # print 'Size MW Img: %d'% max_size_0
#     # print 'Size MH Img: %d'% max_size_1
#     # print 'Starting Point Img: %d'% point
#     # print 'Max Ratio New Img: %d'%max_ratio
#     if(max_ratio==1):
#         if xory:
#             # print "x point"
#             padding = max( (max_size_0 - size_img_0) / 2, 0 )
#         else: 
#             # print "y point"
#             padding = max( (max_size_1 - size_img_1) / 2, 0 )
#         point = point - padding
#     else:   
#         new_img_0=int(size_img_0/max_ratio)
#         new_img_1=int(size_img_1/max_ratio)
#         new_ratio=int(new_img_0/new_img_1)
#         old_ratio=int(size_img_0/size_img_1)
#         if new_ratio is not old_ratio:
#             print "Ratio Error %d : %d"%(new_ratio,old_ratio)
#         if xory:
#             # print "x point"
#             padding = max( (max_size_0 - new_img_0) / 2, 0 )
#         else:
#             # print "y point"
#             padding = max( (max_size_1 - new_img_1) / 2, 0 )
#         point = int(point*max_ratio)- padding
#     # print 'Padding Point Img: %d'%padding 
#     # print 'Ending Point Img: %d'%point
#     return point

def get_orig_point(size_0_orig, size_1_orig, size_0_trasf, size_1_trasf, point, xory):
    orig_ratio=float(size_0_orig)/float(size_1_orig)
    print size_0_orig
    print size_1_orig
    new_ratio=-1
    max_ratio=float(max(float(size_0_orig/size_0_trasf),float(size_1_orig/size_1_trasf),1.0))
    # print 'Size W Img: %d'% size_img_0
    # print 'Size H Img: %d'% size_img_1
    # print 'Size MW Img: %d'% max_size_0
    # print 'Size MH Img: %d'% max_size_1
    # print 'Starting Point Img: %d'% point
    # print 'Max Ratio New Img: %d'%max_ratio
    if(max_ratio==1):
        if xory:
            # print "x point"
            padding = max( (size_0_trasf - size_0_orig) / 2.0, 0.)
        else: 
            # print "y point"
            padding = max( (size_1_trasf - size_1_orig) / 2.0, 0.)
        point = point - padding
        if xory:
            if(point < 0):
                point = 0
            if(point > size_0_orig):
                point= size_0_orig
        else:
            if(point < 0):
                point = 0
            if(point > size_1_orig):
                point= size_1_orig
    else:   
        new_img_0=float(size_0_orig/max_ratio)
        print new_img_0
        new_img_1=float(size_1_orig/max_ratio)
        print new_img_1
        new_ratio=float(new_img_0/new_img_1)
        if new_ratio == orig_ratio:
            if xory:
                # print "x point"
                padding = max( (size_0_trasf - new_img_0) / 2.0, 0.)
            else:
                # print "y point"
                padding = max( (size_1_trasf - new_img_1) / 2.0, 0.)
            point = float(point - padding)*max_ratio
            if xory:
                if(point < 0):
                    point = 0
                if(point > size_0_orig):
                    point= size_0_orig
            else:
                if(point < 0):
                    point = 0
                if(point > size_1_orig):
                    point= size_1_orig
        else: print "Ratio Error Old %.2f : New %.2f"%(new_ratio,orig_ratio)
    # print 'Padding Point Img: %d'%padding 
    # print 'Ending Point Img: %d'%point
    return point

def transform_rect(size_img_0, size_img_1, max_size_0, max_size_1, x1point, y1point, x2point, y2point):
    
    newx1=transform_point(size_img_0, size_img_1, max_size_0, max_size_1, x1point, True)
    newy1=transform_point(size_img_0, size_img_1, max_size_0, max_size_1, y1point, False)
    newx2=transform_point(size_img_0, size_img_1, max_size_0, max_size_1, x2point, True)
    newy2=transform_point(size_img_0, size_img_1, max_size_0, max_size_1, y2point, False)

    return (newx1,newy1,newx2,newy2)

def get_orig_rect(size_img_0, size_img_1, max_size_0, max_size_1,x1point, y1point, x2point, y2point):
    
    newx1=get_orig_point(size_img_0, size_img_1, max_size_0, max_size_1, x1point, True)
    newy1=get_orig_point(size_img_0, size_img_1, max_size_0, max_size_1, y1point, False)
    newx2=get_orig_point(size_img_0, size_img_1, max_size_0, max_size_1, x2point, True)
    newy2=get_orig_point(size_img_0, size_img_1, max_size_0, max_size_1, y2point, False)

    return (newx1,newy1,newx2,newy2)

### Functions for colors managing

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):

    closest_name = closest_colour(requested_colour)

    return closest_name

def get_dominant_color(file_path):

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = 4)
    clt.fit(image)
    hist = centroid_histogram(clt)
    return hist, (get_colour_name(clt.cluster_centers_[0]),get_colour_name(clt.cluster_centers_[1]),get_colour_name(clt.cluster_centers_[2]),get_colour_name(clt.cluster_centers_[3]))

def isnotBlack(file_path):
    percentages, colors = get_dominant_color(file_path)
    tot_black=0.0
    for i in range(0,len(colors)):
        if colors[i] in ['black']:
            # print tot_black
            tot_black=tot_black+percentages[i]
    if(tot_black>=0.9):
        # print("Is black")
        return False
    else: 
        # print("Is not black")
        return True

