import frame as fm
import multiclass_rectangle
import Utils
import progressbar
import os
import vid_classes
from xml.etree import ElementTree
import utils_image

def parse_XML_to_data(xml_list_video):
    frames_list=[]
    video_list=[]
    # image_multi_class= None
    # rectangle_multi = None
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    for i in progress(range(0, len(xml_list_video))):
        # print "Iterating on Video:"+ str(xml_list_video[i][0][0])
        for j in range(0, len(xml_list_video[i])):
            # print "Iterating on Frame:"+ str(xml_list_video[i][j][0])
            with open(xml_list_video[i][j][0], 'rt') as f:
                tree = ElementTree.parse(f)
                for obj in tree.findall('object'):                
                    name = obj.find('name').text
                    class_code= name
                    name = vid_classes.code_to_class_string(name)
                    if name in ["nothing"]:
                        continue
                    else:
                        #The files with the original data path are made in both: multiclass e single class
                        jump=0
                        image_multi_class= fm.Frame_Info()
                        image_multi_class.frame= xml_list_video[i][j][1]
                        # print image_multi_class.frame
                        rectangle_multi= multiclass_rectangle.Rectangle_Multiclass()
                        for node in tree.iter():
                            tag=str(node.tag)
                            if tag in ['name']:
                                if str(vid_classes.code_to_class_string(str(node.text))) in ["nothing"]:
                                    jump = 1
                                else : 
                                    jump=0
                                    rectangle_multi.label_chall=int(vid_classes.class_string_to_comp_code(str(vid_classes.code_to_class_string(str(node.text)))))
                                    # print rectangle_multi.label_chall
                                    rectangle_multi.label_code=str(node.text)
                                    rectangle_multi.label=vid_classes.code_to_class_string(str(node.text))                                
                            if tag in ["xmax"]:
                                if jump == 0:
                                    rectangle_multi.x2=float(node.text)
                            if tag in ["xmin"]:
                                if jump == 0:
                                    rectangle_multi.x1=float(node.text)
                            if tag in ["ymax"]:
                                if jump == 0:
                                    rectangle_multi.y2=float(node.text)                            
                            if tag in ["ymin"]:
                                if jump == 0:    
                                    rectangle_multi.y1=float(node.text)
                                    image_multi_class.append_rect(rectangle_multi)
                        if jump == 0:
                            image_multi_class.append_labeled_rect(rectangle_multi)
                        break
                frames_list.append(image_multi_class)
        video_list.append(frames_list)
        # frames_list=None
        # frames_list=[]        
    return video_list

def read_xml_files(filename , data_folder):
    xml_list_video = []
    video_frame =[]
    with open(filename) as f:
        video=None
        for line in f:
            file, idx = line.strip().split(' ')
            # new_file='ILSVRC2016_'+file.split('_')[1]+'_'+file.split('_')[2]
            # #file.replace('ILSVRC2015','ILSVRC2016')
            # file=new_file
            # print 'File name:%s'%file    
            if video is None:
                video_frame.append((data_folder + file + '.xml',idx))
                video=file.split('/')[0]
            else:
                if video==file.split('/')[0]:
                    video_frame.append((data_folder + file + '.xml',idx))
                else :
                    xml_list_video.append(video_frame)
                    video_frame = []
                    video_frame.append((data_folder + file + '.xml',idx))
                    video=file.split('/')[0]
        xml_list_video.append(video_frame)
    return xml_list_video

def val_to_data(source):
    text_lines=[]
    frames_list=[]
    frame = None
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    with open(source, 'r') as s: 
        for line in s:
            id_frame, id_class, conf, xmin, ymin, xmax, ymax = line.strip().split(' ')
            text_lines.append((id_frame, id_class, conf, xmin, ymin, xmax, ymax))
    for i in range(0, len(text_lines)):
        if frame is None:
            frame = fm.Frame_Info()
            frame.frame= text_lines[i][0]
            rect= multiclass_rectangle.Rectangle_Multiclass()
            # Not all the inserted values are really used
            rect.load_labeled_rect(0, text_lines[i][2], text_lines[i][2], text_lines[i][3], text_lines[i][4], text_lines[i][5], text_lines[i][6], text_lines[i][1], text_lines[i][1], text_lines[i][1])
            frame.append_labeled_rect(rect)
        else :
            if frame.frame == text_lines[i][0]:
                rect= multiclass_rectangle.Rectangle_Multiclass()
                # Not all the inserted values are really used
                rect.load_labeled_rect(0, text_lines[i][2], text_lines[i][2], text_lines[i][3], text_lines[i][4], text_lines[i][5], text_lines[i][6], text_lines[i][1], text_lines[i][1], text_lines[i][1])
                frame.append_labeled_rect(rect)
            else :
                frames_list.append(frame)
                frame = fm.Frame_Info()
                frame.frame= text_lines[i][0]
                rect= multiclass_rectangle.Rectangle_Multiclass()
                # Not all the inserted values are really used
                rect.load_labeled_rect(0, text_lines[i][2], text_lines[i][2], text_lines[i][3], text_lines[i][4], text_lines[i][5], text_lines[i][6], text_lines[i][1], text_lines[i][1], text_lines[i][1])
                frame.append_labeled_rect(rect)
    frames_list.append(frame)
    return frames_list

def parse_video_to_framelist(video_list):

    frames_list=[]
    for video in video_list:
        for frame in video:
            frames_list.append(frame)
    return frames_list

def save_best_overlap(val_bbox, output_bbox):

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    count_best_bbox=0
    len_val_bbox=len(val_bbox)
    len_output_bbox=len(output_bbox)
    count_missing_boxes=0
    with open("best_overlap.txt", 'a') as d:
        for i in progress(range(0, len(val_bbox))):
            for rect in val_bbox[i].rects:
                if(len(output_bbox[i].rects)>0):
                    selected=multiclass_rectangle.pop_max_overlap(output_bbox[i].rects,rect)
                    count_best_bbox=count_best_bbox+1
                    d.write(str(val_bbox[i].frame)+' '+str(rect.label_chall)+ ' 0.5 '+str(selected.x1)+' '+str(selected.y1)+' '+str(selected.x2)+' '+str(selected.y2) + os.linesep)
                else:
                    count_missing_boxes=count_missing_boxes+1
    print "Total Frame Number: "+ str(len_val_bbox) 
    print "Total Output Bounding Boxes: "+ str(len_output_bbox) 
    print "Total Best Bounding Boxes: "+ str(count_best_bbox) 
    print "Total Missing Bounding Boxes: "+ str(count_missing_boxes) 
    print "Total False Positive Bounding Boxes: "+ str(len_output_bbox-count_best_bbox) 
    print "BBox/Frame Number: "+ str(float(count_best_bbox)/float(len_val_bbox)) 
    print "Missing BBox/Frame Number: "+ str(float(float(count_missing_boxes)/float(len_val_bbox)))
    print "False Positive BBox/Frame Number: "+ str(float(float(len_output_bbox-count_best_bbox)/float(len_val_bbox)))

def save_best_iou(val_bbox, output_bbox):

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    count_best_bbox=0
    len_val_bbox=len(val_bbox)
    len_output_bbox=len(output_bbox)
    count_missing_boxes=0
    with open("best_iou.txt", 'a') as d:
        for i in progress(range(0, len(val_bbox))):
            for rect in val_bbox[i].rects:
                if(len(output_bbox[i].rects)>0):
                    selected=multiclass_rectangle.pop_max_iou(output_bbox[i].rects,rect)
                    count_best_bbox=count_best_bbox+1
                    d.write(str(val_bbox[i].frame)+' '+str(rect.label_chall)+ ' 0.5 '+str(selected.x1)+' '+str(selected.y1)+' '+str(selected.x2)+' '+str(selected.y2) + os.linesep)
                else:
                    count_missing_boxes=count_missing_boxes+1
    print "Total Frame Number: "+ str(len_val_bbox) 
    print "Total Output Bounding Boxes: "+ str(len_output_bbox) 
    print "Total Best Bounding Boxes: "+ str(count_best_bbox) 
    print "Total Missing Bounding Boxes: "+ str(count_missing_boxes) 
    print "Total False Positive Bounding Boxes: "+ str(len_output_bbox-count_best_bbox) 
    print "BBox/Frame Number: "+ str(float(count_best_bbox)/float(len_val_bbox)) 
    print "Missing BBox/Frame Number: "+ str(float(float(count_missing_boxes)/float(len_val_bbox)))
    print "False Positive BBox/Frame Number: "+ str(float(float(len_output_bbox-count_best_bbox)/float(len_val_bbox)))


def main():
    xml_file_list = read_xml_files('val.txt', 'val/')
    parsed_xml=parse_XML_to_data(xml_file_list)
    parsed_frames=parse_video_to_framelist(parsed_xml)
    # for xml in parsed_xml:
    #     for frame in xml:
    #         print frame
    val_video_info = val_to_data('output.txt')
    # print val_video_info
    save_best_iou(val_video_info,parsed_frames)



if __name__ == '__main__':
    main()