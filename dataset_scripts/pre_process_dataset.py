
from xml.etree import ElementTree
import os
import shutil
import argparse
import time
from PIL import Image, ImageChops
import progressbar
import glob
from frame import Frame_Info
from multiclass_rectangle import Rectangle_Multiclass
import utils_image
import Utils
import vid_classes
from vid_classes import Classes_List as CL

## SIZE VARIABLES

width=640
height=480

##### GENERAL FUNCTIONS


def get_ordered_name_XML(path_bbox_dataset):

    a = []
    for s in os.listdir(path_bbox_dataset):
         if os.path.isfile(os.path.join(path_bbox_dataset, s)) & s.endswith('.xml'):
            a.append(os.path.join(path_bbox_dataset, s))
    a.sort(key=lambda s: s)
    return a

def create_folder_structure(path_dataset): #Create All the files needed for the New Dataset
    
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)
        print("Created Folder: %s"%path_dataset)
    if not os.path.exists(path_dataset+'/annotations'):
        os.makedirs(path_dataset+'/annotations')
        print("Created Folder: %s"%(path_dataset+'/annotations'))
    if not os.path.exists(path_dataset+'/data'):
        os.makedirs(path_dataset+'/data')
        print("Created Folder: %s"%(path_dataset+'/data'))
    
    for class_code in CL.class_code_string_list:
        if not os.path.exists(path_dataset+'/annotations/'+class_code):
            os.makedirs(path_dataset+'/annotations/'+class_code)
            print("Created Folder: %s"%(path_dataset+'/annotations/'+class_code))
        if not os.path.exists(path_dataset+'/data/'+class_code):
            os.makedirs(path_dataset+'/data/'+class_code)
            print("Created Folder: %s"%(path_dataset+'/data/'+class_code))

    if not os.path.exists(path_dataset+'/dataset_summary.txt'):
        open(path_dataset+'/dataset_summary.txt', 'a')
        print "Created File: %s"%(path_dataset+'/dataset_summary.txt')

def pre_process_dataset(bb_XML_file_list, path_val_folder, path_dataset):

    with open(path_dataset+'/dataset_summary.txt', 'w') as summary:
        for class_code in CL.class_code_string_list:
            start_string="Starting making files for class code: %s ; name: %s, May take a while.... "%(class_code, vid_classes.code_to_class_string(class_code))
            staring_xml_string="Strating xml files: %d"%len(bb_XML_file_list)
            print start_string+ '/n', staring_xml_string+'/n'
            summary.write(start_string+ os.linesep+staring_xml_string+ os.linesep)
            progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
            tot_rect_class=0
            count_xml=0
            bb_folder_path=path_dataset+'/annotations/'+class_code+'/'
            data_folder_path=path_dataset+'/data/'+class_code+'/'
            for file_name in progress(bb_XML_file_list): 
                removed=0            
                delete=False
                with open(file_name, 'rt') as f:
                    #print "File Opened: %s"%file_name
                    tree_new = ElementTree.parse(f)
                    root_new = tree_new.getroot()
                    parent_map_new = dict((c, p) for p in tree_new.getiterator() for c in p)
                    count_rect=0
                    # list so that we don't mess up the order of iteration when removing items.
                    for obj in list(tree_new.findall('object')):
                        obj_class_code = obj.find('name').text
                        if obj_class_code == str(class_code):
                            #print "Founded Object: %s"%vid_classes.code_to_class_string(obj_class_code)
                            #count_rect+1 there's an object of that class in the xml file
                            count_rect=count_rect+1
                            delete=True
                        else:
                            #eliminate node
                            #print "Eliminated Node: %s"%vid_classes.code_to_class_string(obj_class_code)
                            parent_map_new[obj].remove(obj)
                            removed=removed+1
                    if count_rect>0:
                        ### Means the file belongs to the class so we change filename and directory name and we copy the image to the dataset
                        for node in tree_new.iter():
                            tag=str(node.tag)
                            if tag in ["folder"]:
                                path_orig_file=path_val_folder+'/'+str(node.text)
                                node.text= data_folder_path
                                # print "Changed folder from: %s to : %s"%(path_orig_file, node.text)
                            if tag in ["filename"]:
                                path_orig_file=path_orig_file+'/'+str(node.text)+'.JPEG'
                                new_filename=class_code+'_'+str(000000+count_xml)+'.JPEG'
                                path_new_file=data_folder_path+new_filename
                                node.text= new_filename
                                # print "Changed Name from: %s to : %s"%(path_orig_file, path_new_file)
                        xml_filename=bb_folder_path+class_code+'_'+str(000000+count_xml)+'.xml'
                        tree_new.write(xml_filename)
                        shutil.copy2(path_orig_file, path_new_file)
                        # print "Saved New .xml file: %s"%xml_filename
                        # print "Saved New .jpeg image: %s"%path_new_file
                        if (removed == 0 ) & delete:
                            os.remove(path_orig_file)
                            # print "Removed image: %s"%(path_orig_file)
                        count_xml=count_xml+1
                        tot_rect_class=tot_rect_class+count_rect
                        ##TODO: Copy Image
                # print "Count_Rect %d"%count_rect
                # print "Removed %d"%removed
                if (removed == 0 ) & delete:
                    os.remove(file_name)
                    # print "Removed XML: %s"%(file_name)
                    bb_XML_file_list.remove(file_name)
                if (removed >0)&(count_rect>0):
                    with open(file_name, 'rt') as f:
                        tree_new = ElementTree.parse(f)
                        root_new = tree_new.getroot()
                        parent_map_new = dict((c, p) for p in tree_new.getiterator() for c in p)
                        for obj in list(tree_new.findall('object')):
                            obj_class_code = obj.find('name').text
                            if obj_class_code == str(class_code):
                                parent_map_new[obj].remove(obj)
                        tree_new.write(file_name)
            end_string="Ended with Success Process for class:%s"%class_code 
            parsed_bb_string="Parsed: %d BB Files"%count_xml
            added_rect_String="Added: %d Object Rectangles"%tot_rect_class
            print end_string+'/n',parsed_bb_string +'/n',added_rect_String+'/n'
            summary.write(end_string+ os.linesep+parsed_bb_string+ os.linesep+added_rect_String+ os.linesep)

def main():
    '''
    Parse command line arguments and execute the code

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./dataset', type=str)
    parser.add_argument('--bb_folder', required=True, type=str)
    parser.add_argument('--val_folder', required=True, type=str)
    args = parser.parse_args()

    
    start = time.time()

    bb_XML_file_list=[]
    create_folder_structure(args.dataset_path)
    bb_XML_file_list= get_ordered_name_XML(args.bb_folder)
    pre_process_dataset(bb_XML_file_list, args.val_folder, args.dataset_path)
    # shutil.rmtree(args.val_folder, ignore_errors=True)
    # shutil.rmtree(args.bb_folder, ignore_errors=True)

    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")


if __name__ == '__main__':
    main()
