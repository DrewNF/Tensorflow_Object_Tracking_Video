import os
import cv2
import progressbar
import Utils_Image

def make_video_from_list(out_vid_path, frames_list):
	if frames_list[0] is not None:
	    img = cv2.imread(frames_list[0], True)
	    print frames_list[0]
	    h, w = img.shape[:2]
	    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	    out = cv2.VideoWriter(out_vid_path,fourcc, 20.0, (w, h), True)
	    print("Start Making File Video:%s " % out_vid_path)
	    print("%d Frames to Compress"%len(frames_list))
	    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
	    for i in progress(range(0,len(frames_list))):
	        if Utils_Image.check_image_with_pil(frames_list[i]):
	            out.write(img)
	            img = cv2.imread(frames_list[i], True)
	    out.release()
	    print("Finished Making File Video:%s " % out_vid_path)


def make_video_from_frames(out_vid_path, frames):
	if frames[0] is not None:
	    h, w = frames[0].shape[:2]
	    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	    out = cv2.VideoWriter(out_vid_path,fourcc, 20.0, (w, h), True)
	    print("Start Making File Video:%s " % out_vid_path)
	    print("%d Frames to Compress"%len(frames))
	    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
	    for i in progress(range(0,len(frames))):
	        out.write(frames[i])
	    out.release()
	    print("Finished Making File Video:%s " % out_vid_path)


####### FOR TENSORBOX ###########

def extract_idl_from_frames(vid_path, video_perc, path_video_folder, folder_path_frames, idl_filename):
    
    ####### Creating Folder for the video frames and the idl file for the list
    
    if not os.path.exists(path_video_folder):
        os.makedirs(path_video_folder)
        print("Created Folder: %s"%path_video_folder)
    if not os.path.exists(path_video_folder+'/'+folder_path_frames):
        os.makedirs(path_video_folder+'/'+folder_path_frames)
        print("Created Folder: %s"% (path_video_folder+'/'+folder_path_frames))
    if not os.path.exists(idl_filename):
        open(idl_filename, 'a')
        print "Created File: "+ idl_filename
    list=[]
    # Opening & Reading the Video

    print("Opening File Video:%s " % vid_path)
    vidcap = cv2.VideoCapture(vid_path)
    if not vidcap.isOpened():
        print "could Not Open :",vid_path
        return
    print("Opened File Video:%s " % vid_path)
    print("Start Reading File Video:%s " % vid_path)
    
    total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
    
    print("%d Frames to Read"%total)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    image = vidcap.read()
    with open(idl_filename, 'w') as f:
        for i in progress(range(0,total)):
            #frame_name="%s/%s/fram%d.jpeg"%(path_video_folder,folder_path_frames,i)
            list.append("%s/%sframe%d.jpeg"%(path_video_folder,folder_path_frames,i))
            cv2.imwrite("%s/%sframe%d.jpeg"%(path_video_folder,folder_path_frames,i), image[1])     # save frame as JPEG file
            image = vidcap.read()

    print("Finish Reading File Video:%s " % vid_path)
    return list


####### FOR YOLO ###########

def extract_frames(vid_path, video_perc):
    list=[]
    frames=[]
    # Opening & Reading the Video
    print("Opening File Video:%s " % vid_path)
    vidcap = cv2.VideoCapture(vid_path)
    if not vidcap.isOpened():
        print "could Not Open :",vid_path
        return
    print("Opened File Video:%s " % vid_path)
    print("Start Reading File Video:%s " % vid_path)
    image = vidcap.read()
    total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
    print("%d Frames to Read"%total)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    for i in progress(range(0,total)):
        list.append("frame%d.jpg" % i)
        frames.append(image)
        image = vidcap.read()
    print("Finish Reading File Video:%s " % vid_path)
    return frames, list
