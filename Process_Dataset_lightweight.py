
from xml.etree import ElementTree
import os
import shutil
import argparse
import time
from PIL import Image, ImageChops
import progressbar
from Utils_Picture import Picture_Info, BB_Rectangle
import Utils_Image
import Utils
import Classes
from Classes import Classes_List as CL

##### GENERAL VARIABLES #######

##MULTICLASS NEW DATASET

string_mltcl_bb_file = 'mltcl_bb_file_list.txt'
string_mltcl_class_code_file = 'mltcl_class_code_file_list.txt'
string_mltcl_class_name_file = 'mltcl_class_name_file_list.txt'
string_mltcl_chall_code_file = 'mltcl_chall_code_file_list.txt'

##SINGLE CLASS NEW DATASET

string_bb_file = '_bb_file_list.txt'
string_class_code_file = '_class_code_file_list.txt'
string_class_name_file = '_class_name_file_list.txt'
string_chall_code_file = '_chall_code_file_list.txt'

## SIZE VARIABLES

width=640
height=480

##### GENERAL FUNCTIONS

def create_summary_files(path_dataset): #Create All the files needed for the New Dataset
    
    path_mltcl_bb_file= path_dataset+'/'+string_mltcl_bb_file
    path_mltcl_class_code_file= path_dataset+'/'+string_mltcl_class_code_file
    path_mltcl_class_name_file= path_dataset+'/'+string_mltcl_class_name_file
    path_mltcl_chall_code_file= path_dataset+'/'+string_mltcl_chall_code_file

    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)
        print("Created Folder: %s"%path_dataset)
    if not os.path.exists(path_mltcl_bb_file):
        open(path_mltcl_bb_file, 'a')
        print "Created File: "+ path_mltcl_bb_file
    if not os.path.exists(path_mltcl_class_code_file):
        open(path_mltcl_class_code_file, 'a')
        print "Created File: "+ path_mltcl_class_code_file
    if not os.path.exists(path_mltcl_class_name_file):
        open(path_mltcl_class_name_file, 'a')
        print "Created File: "+ path_mltcl_class_name_file
    if not os.path.exists(path_mltcl_chall_code_file):
        open(path_mltcl_chall_code_file, 'a')
        print "Created File: "+ path_mltcl_chall_code_file
    
    for class_name in CL.class_name_string_list:


        path_bb_file= path_dataset+'/'+class_name+'/'+class_name+string_bb_file
        path_class_code_file= path_dataset+'/'+class_name+'/'+class_name+string_class_code_file
        path_class_name_file= path_dataset+'/'+class_name+'/'+class_name+string_class_name_file
        path_chall_code_file= path_dataset+'/'+class_name+'/'+class_name+string_chall_code_file
        

        if not os.path.exists(path_dataset+'/'+class_name):
            os.makedirs(path_dataset+'/'+class_name)
            print("Created Folder: %s"%(path_dataset+'/'+class_name))
        
        if not os.path.exists(path_bb_file):
            open(path_bb_file, 'a')
            print "Created File: "+ path_bb_file
        if not os.path.exists(path_class_code_file):
            open(path_class_code_file, 'a')
            print "Created File: "+ path_class_code_file
        if not os.path.exists(path_class_name_file):
            open(path_class_name_file, 'a')
            print "Created File: "+ path_class_name_file
        if not os.path.exists(path_chall_code_file):
            open(path_chall_code_file, 'a')
            print "Created File: "+ path_chall_code_file

def parse_XML_lightweight_txt(bb_XML_file_list, path_val_folder, path_dataset):

    count_rect = 0
    count_img = 0

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    print "Start Processing & Building Dataset... may take a while..."

    path_mltcl_bb_file=path_dataset+'/'+string_mltcl_bb_file # Create this file in .dataset/airplane/airplane_bb_mltcl_file_list.txt
    path_mltcl_class_code_file=path_dataset+'/'+string_mltcl_class_code_file
    path_mltcl_class_name_file=path_dataset+'/'+string_mltcl_class_name_file
    path_mltcl_chall_code_file=path_dataset+'/'+string_mltcl_chall_code_file

    for file_name in progress(bb_XML_file_list):
        with open(file_name, 'rt') as f:
            tree = ElementTree.parse(f)
            for obj in tree.findall('object'):
                name = obj.find('name').text
                class_code= name
                name = Classes.code_to_class_string(name)

                if name in ["nothing"]:
                    continue
                else:
                    
                    same_label=0
                    #The files with the original data path are made in both: multiclass e single class
                    
                    
                    path_bb_file=path_dataset+'/'+name+'/'+ name+string_bb_file
                    path_class_code_file= path_dataset+'/'+name+'/'+name+string_class_code_file
                    path_class_name_file= path_dataset+'/'+name+'/'+name+string_class_name_file
                    path_chall_code_file= path_dataset+'/'+name+'/'+name+string_chall_code_file


                    path_orig_file=path_val_folder

                    
                    jump=0
                    
                    image_single_class= Picture_Info()
                    image_single_class.dataset_path= path_val_folder

                    image_multi_class= Picture_Info()
                    image_multi_class.dataset_path= path_val_folder


                    rectangle_single= BB_Rectangle()
                    rectangle_multi= BB_Rectangle()
                    
                    #xmin x1 letf
                    #ymin y1 bottom
                    #xmax x2 right
                    #ymax y2 top
                
                    for node in tree.iter():
                        tag=str(node.tag)
            
                        if tag in ["folder"]:
                            path_orig_file=path_orig_file+'/'+str(node.text)
                            image_single_class.folder= str(node.text)                            
                            image_multi_class.folder= str(node.text)

                        if tag in ["filename"]:
                            image_single_class.filename=str(node.text)+'.PNG'
                            image_multi_class.filename=str(node.text)+'.PNG'

                            path_orig_file=path_orig_file+'/'+str(node.text)+'.JPEG'

                        if tag in ['name']:
                            if str(Classes.code_to_class_string(str(node.text))) in ["nothing"]:
                                jump = 1
                            else : 
                                jump=0
                                rectangle_multi.label_chall=int(Classes.class_string_to_comp_code(str(Classes.code_to_class_string(str(node.text)))))
                                rectangle_multi.label_code=str(node.text)
                                rectangle_multi.label=Classes.code_to_class_string(str(node.text))

                                if str(node.text) == class_code: 
                                    same_label = 1
                                    rectangle_single.label_chall=int(Classes.class_string_to_comp_code(str(Classes.code_to_class_string(str(node.text)))))
                                    rectangle_single.label_code=str(node.text)
                                    rectangle_single.label=Classes.code_to_class_string(str(node.text))
                                else: same_label = 0
                                
                        if tag in ["xmax"]:
                            if jump == 0:
                                rectangle_multi.x2=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                                if same_label==1:
                                    rectangle_single.x2=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                        if tag in ["xmin"]:
                            if jump == 0:
                                rectangle_multi.x1=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                                if same_label==1:
                                    rectangle_single.x1=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                        if tag in ["ymax"]:
                            if jump == 0:
                                rectangle_multi.y2=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))                            
                                image_multi_class.append_rect(rectangle_multi) 
                                count_rect=count_rect+1
                                if same_label==1:
                                    rectangle_single.y2=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                                    image_single_class.append_rect(rectangle_single)
                        if tag in ["ymin"]:
                            if jump == 0:    
                                rectangle_multi.y1=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))
                                if same_label==1:
                                    rectangle_single.y1=float(Utils_Image.transform_point(image_multi_class.width,image_multi_class.height,width, height,float(node.text), False))

                    if jump == 0:
                        
                        count_img=count_img+1

                        out_stream = open(path_mltcl_bb_file, "a")
                        out_stream.write(image_multi_class.get_info_string()+ os.linesep)
                        
                        out_stream = open(path_mltcl_class_code_file, "a")
                        out_stream.write(image_multi_class.get_rects_code()+ os.linesep)
                        
                        out_stream = open(path_mltcl_class_name_file, "a")
                        out_stream.write(image_multi_class.get_rects_labels()+ os.linesep)
                        
                        out_stream = open(path_mltcl_chall_code_file, "a")
                        out_stream.write(image_multi_class.get_rects_chall() + os.linesep)

                        if same_label==1:
                            out_stream = open(path_bb_file, "a")
                            out_stream.write(image_single_class.get_info_string()+ os.linesep)
                            
                            out_stream = open(path_class_code_file, "a")
                            out_stream.write(image_single_class.get_rects_chall()+ os.linesep)
                            
                            out_stream = open(path_class_name_file, "a")
                            out_stream.write(image_single_class.get_rects_labels()+ os.linesep)
                            
                            out_stream = open(path_chall_code_file, "a")
                            out_stream.write(image_single_class.get_rects_code() + os.linesep)
                    break
    print "SUMMARY:"
    print "Parsed: %d BB Files"%len(bb_XML_file_list)
    print "Added: %d Object Rectangles"%count_rect
    print "Added: %d Images"%count_img
    if count_img>0: 
        print "Ratio BB_Files/Images: %.2f"%(float(count_img)/float(len(bb_XML_file_list)))
    if count_rect>0:
        print "Ratio Images/Object_Rectangles: %.2f"%(float(count_img)/float(count_rect))

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
    create_summary_files(args.dataset_path)
    bb_XML_file_list= Utils.get_Files_List(args.bb_folder)
    parse_XML_lightweight_txt(bb_XML_file_list, args.val_folder, args.dataset_path)  

    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")


if __name__ == '__main__':
    main()


