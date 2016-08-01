
import os
import time
from PIL import Image, ImageChops
import progressbar
import argparse
import Utils_Image

##### MAIN ############

def main():
    '''
    Parse command line arguments and execute the code
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--newext', default='.PNG', type=str)
    parser.add_argument('--oldext', default='.JPEG', type=str)
    args = parser.parse_args()

    start = time.time()

    image_list= Utils_Image.get_Image_List(args.dataset_path, args.oldext)

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    print "Start Processing... May take a while..."

    for image_path in progress(image_list):
        Utils_Image.resizeImage(image_path)
        Utils_Image.change_extension(image_path,args.oldext,args.newext)
 
    end = time.time()
    print("Parsed: %d Image of the Dataset"%(len(image_list)))
    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")


if __name__ == '__main__':
    main()