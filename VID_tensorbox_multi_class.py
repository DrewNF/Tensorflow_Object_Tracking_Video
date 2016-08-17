#### My import

import argparse
import utils_image
import utils_video
import Utils_Tensorbox
import Utils_Imagenet
import frame
import vid_classes
import progressbar
import time
import os

######### MAIN ###############

def main():
    '''
    Parse command line arguments and execute the code 

    '''

    ######### TENSORBOX PARAMETERS


    start = time.time()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--result_folder', default='summary_result/', type=str)
    # parser.add_argument('--summary_file', default='results.txt', type=str)
    parser.add_argument('--output_name', default='output.mp4', type=str)
    parser.add_argument('--hypes', default='./TENSORBOX/hypes/overfeat_rezoom.json', type=str)
    parser.add_argument('--weights', default='./TENSORBOX/data/save.ckpt-1250000', type=str)
    parser.add_argument('--perc', default=2, type=int)
    parser.add_argument('--path_video', default='ILSVRC2015_val_00013002.mp4', type=str)# required=True, type=str)

    args = parser.parse_args()

    # hypes_file = './hypes/overfeat_rezoom.json'
    # weights_file= './output/save.ckpt-1090000'

    path_video_folder = os.path.splitext(os.path.basename(args.path_video))[0]
    pred_idl = './%s/%s_val.idl' % (path_video_folder, path_video_folder)
    idl_filename=path_video_folder+'/'+path_video_folder+'.idl'
    frame_list=[]
    frame_list = utils_video.extract_idl_from_frames(args.path_video, args.perc, path_video_folder, 'frames/', idl_filename )

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    for image_path in progress(frame_list):
        utils_image.resizeImage(image_path)
    utils_image.resizeImage(-1)

    video_info=Utils_Tensorbox.bbox_det_TENSORBOX_multiclass( frame_list, path_video_folder, args.hypes, args.weights, pred_idl)
    tracked_video=utils_video.track_objects(video_info)
    frame.saveVideoResults(idl_filename,tracked_video)
    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")

if __name__ == '__main__':
    main()




