#### Import from Tensorbox Project

import tensorflow as tf
import json
import subprocess
from scipy.misc import imread
import numpy as np

# Import DET Alg package

# import sys
# sys.path.insert(0, 'TENSORBOX')

# Original
from utils import googlenet_load, train_utils
from utils.annolist import AnnotationLib as al
from utils.rect import Rect
#Modified

from rect_multiclass import Rect_Multiclass


#### My import

import Classes
import Utils_Image
import Utils_Video
import progressbar
import os




folder_path_det_frames ='det_frames/'
folder_path_det_result='det_results/'
folder_path_frames='frames/'

####### FUNCTIONS DEFINITIONS

def NMS(rects,overlapThresh=0.2):
		# if there are no boxes, return an empty list
	if len(rects) == 0:
		print "WARNING: Passed Empty Boxes Array"
		return []
 
	# initialize the list of picked indexes
	pick = []
	x1, x2, y1, y2=[],[],[],[]
 	for rect in rects:
 		x1.append(rect.x1)
 		x2.append(rect.x2)
 		y1.append(rect.y1)
 		y2.append(rect.y2)
	# grab the coordinates of the bounding boxes
	x1 = np.array(x1)
	y1 = np.array(y1)
	x2 = np.array(x2)
	y2 = np.array(y2)
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
	picked =[]
	for i in pick: picked.append(rects[i])
	return picked

def getTextIDL(annotations):

	frame = -1
	conf=0
	silhouette=-1
	xmin,ymin,xmax,ymax=0,0,0,0

	detections_array=[]

	if annotations.frameNr is not -1:
		frame=annotations.frameNr
	for rect in annotations.rects:
		if rect.silhouetteID is not -1:
			silhouette=rect.silhouetteID
		conf = rect.score
		xmin,ymin,xmax,ymax = rect.x1,rect.y1,rect.x2 ,rect.y2
		detections_array.append(str(frame)+' '+str(silhouette)+' '+str(conf)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax))
	return detections_array

def writeText(annotations, file):
	detections= getTextIDL(annotations)
	for detection in detections:
		file.write(detection + os.linesep)

def saveTextResults(filename, annotations):
    if not os.path.exists(filename):
        print "Created File: "+ filename
    file = open(filename, 'w')
    for annotation in annotations:
        writeText(annotation,file)
    file.close()

def get_silhouette_confidence(silhouettes_confidence):
    higher=0.0
    index=0
    # print "conf_sil : " + str(silhouettes_confidence)
    # print "conf_sil LEN : " + str(len(silhouettes_confidence))

    for i in range(0,len(silhouettes_confidence)):
        # print "conf_sil I : " + str(silhouettes_confidence[i])
        if silhouettes_confidence[i]>higher:
            higher = silhouettes_confidence[i]
            index = i
    # print str(index+1),str(higher)
    return index+1 , higher

def get_higher_confidence(rectangles):
    higher=0.0
    index=0
    # print "conf_sil : " + str(silhouettes_confidence)
    # print "conf_sil LEN : " + str(len(silhouettes_confidence))

    for rect in rectangles:
        # print "conf_sil I : " + str(silhouettes_confidence[i])
        if rect.true_confidence>higher:
            higher = rect.true_confidence
    # print str(index+1),str(higher)
    return  higher-0.1

def print_logits(logits):
    higher=0.0
    index=0
    # print "logits_sil shape : " + str(logits.shape)

    for i in range(0,len(logits)):
        # print "conf_sil I : " + str(logits[i])
        for j in range(0,len(logits[i])):
            if logits[i][0][j]>higher:
                higher = logits[i][0][j]
                index = j
    print str(index+1),str(higher)
    return index+1 , higher

def get_singleclass_rectangles(H, confidences, boxes, arch, min_conf,rnn_len=1):
    boxes_r = np.reshape(boxes, (-1,
                                 arch["grid_height"],
                                 arch["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             arch["grid_height"],
                                             arch["grid_width"],
                                             rnn_len,
                                             2))
    cell_pix_size = H['arch']['region_size']
    all_rects = [[[] for _ in range(arch["grid_width"])] for _ in range(arch["grid_height"])]

    for n in range(0, H['arch']['rnn_len']):
        for y in range(arch["grid_height"]):
            for x in range(arch["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                conf = confidences_r[0, y, x, n, 1]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                h = max(1, bbox[3])
                w = max(1, bbox[2])
                #w = h * 0.4
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
    all_rects_r = [r for row in all_rects for cell in row for r in cell if r.true_confidence > min_conf]
    rects = []
    for rect in all_rects_r:
        r = al.AnnoRect()
        r.x1 = rect.cx - rect.width/2.
        r.x2 = rect.cx + rect.width/2.
        r.y1 = rect.cy - rect.height/2.
        r.y2 = rect.cy + rect.height/2.
        r.score = rect.true_confidence
        rects.append(r)
    
    return rects

def get_multiclass_rectangles(H, confidences, boxes, arch, rnn_len):
    boxes_r = np.reshape(boxes, (-1,
                                 arch["grid_height"],
                                 arch["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             arch["grid_height"],
                                             arch["grid_width"],
                                             rnn_len,
                                             H['arch']['num_classes']))
    # print "boxes_r shape" + str(boxes_r.shape)
    # print "confidences" + str(confidences.shape)
    cell_pix_size = H['arch']['region_size']
    all_rects = [[[] for _ in range(arch["grid_width"])] for _ in range(arch["grid_height"])]
    for n in range(rnn_len):
        for y in range(arch["grid_height"]):
            for x in range(arch["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                # conf = np.max(confidences_r[0, y, x, n, 1:])
                index, conf = get_silhouette_confidence(confidences_r[0, y, x, n, 1:])
                # print index, conf
                # print np.max(confidences_r[0, y, x, n, 1:])
                # print "conf" + str(conf)
                # print "conf" + str(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect_Multiclass(abs_cx,abs_cy,w,h,conf, index))
    # print "confidences_r" + str(confidences_r.shape)

    all_rects_r = [r for row in all_rects for cell in row for r in cell]
    min_conf = get_higher_confidence(all_rects_r)
    rects = []
    for rect in all_rects_r:
    	if rect.true_confidence>min_conf:
	        r = al.AnnoRect()
	        r.x1 = rect.cx - rect.width/2.
	        r.x2 = rect.cx + rect.width/2.
	        r.y1 = rect.cy - rect.height/2.
	        r.y2 = rect.cy + rect.height/2.
	        r.score = rect.true_confidence
	        r.silhouetteID=rect.silhouette
	        rects.append(r)
    return rects

def draw_rectangles(orig_img, save_img, rects):

    from PIL import Image,ImageDraw


    bb_img = Image.open(orig_img)
    # print orig_img
    for bb_rect in rects:
    ################ Adding Rectangle ###################
        dr = ImageDraw.Draw(bb_img)
        cor = (bb_rect.x1,bb_rect.y1,bb_rect.x2 ,bb_rect.y2) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
        if bb_rect.silhouetteID is  -1:
            outline_class=(240,255,240)
        else :  
            outline_class=Classes.code_to_color(bb_rect.silhouetteID)
        dr.rectangle(cor, outline=outline_class)
    # print save_img  
    bb_img.save(save_img)

def still_image_TENSORBOX_singleclass(frames_list,path_video_folder,hypes_file,weights_file,pred_idl, min_conf=0.9):
    
    from train import build_forward

    print("Starting DET Phase")
    
    if not os.path.exists(path_video_folder+'/'+folder_path_det_frames):
        os.makedirs(path_video_folder+'/'+folder_path_det_frames)
        print("Created Folder: %s"%path_video_folder+'/'+folder_path_det_frames)
    if not os.path.exists(path_video_folder+'/'+folder_path_det_result):
        os.makedirs(path_video_folder+'/'+folder_path_det_result)
        print("Created Folder: %s"% path_video_folder+'/'+folder_path_det_result)

    det_frames_list=[]

    #### START TENSORBOX CODE ###

    ### Opening Hypes file for parameters
    
    with open(hypes_file, 'r') as f:
        H = json.load(f)

    ### Get Annotation List of all the image to test
    idl_filename=path_video_folder+'/'+path_video_folder+'.idl'

    ### Building Network

    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['arch']['image_height'], H['arch']['image_width'], 3])

    if H['arch']['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        grid_area = H['arch']['grid_height'] * H['arch']['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['arch']['rnn_len'], 2])), [grid_area, H['arch']['rnn_len'], 2])
    if H['arch']['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, weights_file )##### Restore a Session of the Model to get weights and everything working
    
        annolist = al.AnnoList()
    
        #### Starting Evaluating the images
        lenght=int(len(frames_list))
        
        print("%d Frames to DET"%len(frames_list))
        
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        frameNr=0
        skipped=0
        for i in progress(range(0, len(frames_list))):
            # img = Image.open(frames_list[i])
            # if img.getbbox()is not None:
            if Utils_Image.isnotBlack(frames_list[i]) & Utils_Image.check_image_with_pil(frames_list[i]):
                img = imread(frames_list[i])
                feed = {x_in: img}
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

                pred_anno = al.Annotation()        
            
                rects = get_singleclass_rectangles(H, np_pred_confidences, np_pred_boxes,H["arch"],min_conf, rnn_len=H['arch']['rnn_len'])
                pred_anno.rects = rects
                pred_anno.imageName = frames_list[i]
                pred_anno.frameNr = frameNr
                frameNr=frameNr+1
                pick = NMS(rects)
                draw_rectangles(frames_list[i],frames_list[i], pick)

                det_frames_list.append(frames_list[i])            
                annolist.append(pred_anno)
            else: skipped=skipped+1 

    saveTextResults(idl_filename,annolist)
    annolist.save(pred_idl)
    print("Skipped %d Black Frames"%skipped)

    #### END TENSORBOX CODE ###

    return det_frames_list

def still_image_TENSORBOX_multiclass(frames_list,path_video_folder,hypes_file,weights_file,pred_idl):
    
    from train_multiclass import build_forward

    print("Starting DET Phase")
    
    if not os.path.exists(path_video_folder+'/'+folder_path_det_frames):
        os.makedirs(path_video_folder+'/'+folder_path_det_frames)
        print("Created Folder: %s"%path_video_folder+'/'+folder_path_det_frames)
    if not os.path.exists(path_video_folder+'/'+folder_path_det_result):
        os.makedirs(path_video_folder+'/'+folder_path_det_result)
        print("Created Folder: %s"% path_video_folder+'/'+folder_path_det_result)

    det_frames_list=[]

    #### START TENSORBOX CODE ###
    idl_filename=path_video_folder+'/'+path_video_folder+'.idl'

    ### Opening Hypes file for parameters
    
    with open(hypes_file, 'r') as f:
        H = json.load(f)

    ### Building Network

    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['arch']['image_height'], H['arch']['image_width'], 3])

    if H['arch']['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        grid_area = H['arch']['grid_height'] * H['arch']['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['arch']['rnn_len'], H['arch']['num_classes']])), [grid_area, H['arch']['rnn_len'], H['arch']['num_classes']])
        pred_logits = tf.reshape(tf.nn.softmax(tf.reshape(pred_logits, [grid_area * H['arch']['rnn_len'], H['arch']['num_classes']])), [grid_area, H['arch']['rnn_len'], H['arch']['num_classes']])
    if H['arch']['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)

    saver = tf.train.Saver()

    with tf.Session() as sess:


        sess.run(tf.initialize_all_variables())
        saver.restore(sess, weights_file )##### Restore a Session of the Model to get weights and everything working
    
        annolist = al.AnnoList()
    
        #### Starting Evaluating the images
        lenght=int(len(frames_list))
        
        print("%d Frames to DET"%len(frames_list))
        
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        frameNr=0
        skipped=0
        for i in progress(range(0, len(frames_list))):

            if Utils_Image.isnotBlack(frames_list[i]) & Utils_Image.check_image_with_pil(frames_list[i]):

                img = imread(frames_list[i])
                feed = {x_in: img}
                (np_pred_boxes,np_pred_logits, np_pred_confidences) = sess.run([pred_boxes,pred_logits, pred_confidences], feed_dict=feed)
                # print_logits(np_pred_confidences)
                pred_anno = al.Annotation()
                #pred_anno.imageName = test_anno.imageName
            
                # print "np_pred_confidences shape" + str(np_pred_confidences.shape)
                # print "np_pred_boxes shape" + str(np_pred_boxes.shape)
                # for i in range(0, np_pred_confidences.shape[0]):
                #     print np_pred_confidences[i]
                #     for j in range(0, np_pred_confidences.shape[2]):
                #         print np_pred_confidences[i][0][j]

                rects = get_multiclass_rectangles(H, np_pred_confidences, np_pred_boxes,H["arch"], rnn_len=H['arch']['rnn_len'])
                pred_anno.rects = rects
                pred_anno.imageName = frames_list[i]
                pred_anno.frameNr = frameNr
                frameNr=frameNr+1
                det_frames_list.append(frames_list[i])
                pick = NMS(rects)
                draw_rectangles(frames_list[i],frames_list[i], pick)

                annolist.append(pred_anno)

            else: skipped=skipped+1 

        saveTextResults(idl_filename,annolist)
        annolist.save(pred_idl)
        print("Skipped %d Black Frames"%skipped)

    #### END TENSORBOX CODE ###

    return det_frames_list
