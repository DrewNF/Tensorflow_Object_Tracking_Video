#### Import from Tensorbox Project

from utils.annolist import AnnotationLib as al

#### My import

from PIL import Image, ImageChops,ImageDraw
import progressbar
import time
import cv2
import os
import argparse
import sys

######### PARAMETERS

def test_IDL(idl_filename):

    print("Starting Testing Dataset... May take a while")
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    test_annos = al.parse(idl_filename)
    for test_anno in progress(test_annos):
        bb_img = Image.open(test_anno.imageName)
        orig_img = cv2.imread(test_anno.imageName, 0)
        cv2.imshow('Original Image',orig_img)
        cv2.waitKey(2)
        for test_rect in test_anno.rects:
            dr = ImageDraw.Draw(bb_img)
            cor = (test_rect.x2,test_rect.y2,test_rect.x1,test_rect.y1 ) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
            dr.rectangle(cor, outline="green")   
        image_name, image_ext = os.path.splitext(test_anno.imageName) 
        bb_img.save(image_name+'_copy'+image_ext)
        bb_img = cv2.imread(image_name+'_copy'+image_ext, 0)
        cv2.imshow('Mine Rectangle detection',bb_img)
        cv2.waitKey(2)
        os.remove(image_name+'_copy'+image_ext)


######### MAIN ###############


def main():
    '''
    Parse command line arguments and execute the code 

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--idl', required=True, type=str)
    args = parser.parse_args()

    start = time.time()

    test_IDL(args.idl)

    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")

if __name__ == '__main__':
    main()


