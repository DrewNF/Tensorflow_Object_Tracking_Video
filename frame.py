import copy 
import os
import multiclass_rectangle
from multiclass_rectangle import Rectangle_Multiclass
import vid_classes

#####################################################################################################################################################################
########################## CLASS AND FUNCTIONS DEFINED TO USE WITH INCEPTION EXTENDS THE ONES ABOVE AND HAVE THE SAME SCOPE) ########################################
#####################################################################################################################################################################

####  LABELED Picture Informations Class

class Frame_Info(object):
    ### Here the rects are Labeled_BB not BB_Rectangle
    def __init__(self):

        self.num_obj=-1
        self.rects = []
        self.folder='Not Set'
        self.filename='Not Set'
        self.dataset_path='Not Set'
        self.default_path='Not Set'
        self.width=-1
        self.height=-1
        self.frame=-1

    ### Save Copy functions

    def duplicate(self):
        
        new_copy= Frame_Info()
        new_copy.folder=copy.copy(self.folder)
        new_copy.filename=copy.copy(self.filename)
        new_copy.dataset_path=copy.copy(self.dataset_path)
        new_copy.default_path=copy.copy(self.default_path)
        new_copy.width=copy.copy(self.width)
        new_copy.height=copy.copy(self.height)
        new_copy.frame=copy.copy(self.frame)

        return new_copy

    def duplicate_rects(self, rects):
        for rect in rects:
            self.append_labeled_rect(rect.duplicate())

    ### Save insert of rect into the frame rects list

    def append_labeled_rect(self, rectangle):
        """Adding rect to the picture_info."""
        index= len(self.rects)
        self.rects.insert(index, rectangle)

    def append_rect(self, rectangle):
        """Adding rect to the picture_info."""
        rect= Rectangle_Multiclass()
        rect.load_BBox( rectangle.x1, rectangle.x2, rectangle.y1, rectangle.y2, rectangle.label, rectangle.label_chall, rectangle.label_code)
        index= len(self.rects)
        self.rects.insert(index, rect)

    ### Functions for parse the xml into the .idl file to train TENSORBOX

    def get_rects_string(self):
        """Get the string of the coordinates of all the rects of the image."""
        string='"'+self.dataset_path+'/'+self.folder+'/'+self.filename+'"' 
        if self.frame is not -1:
           string= string+ ' @'+ str(self.frame)
        string=string+' : '
        n_obj=len(self.rects)
        for rectangle in self.rects:
            string = string + rectangle.get_rect_string()
            n_obj=n_obj-1
            if n_obj>0:
                string= string + ','
            else: string= string + ';'
        return string

    def get_rects_labels(self):
        """Get the string of the coordinates of all the rects of the image."""
        
        string='"'+self.dataset_path+'/'+self.folder+'/'+self.filename+'"'
        if self.frame is not -1:
           string= string+ ' @'+ str(self.frame)
        string=string+' : '
        n_obj=len(self.rects)
        for rectangle in self.rects:
            string = string + rectangle.get_label_string()
            n_obj=n_obj-1
            if n_obj>0:
                string= string + ','
            else: string= string + ';'
        return string

    def get_rects_code(self):
        """Get the string of the coordinates of all the rects of the image."""
        string='"'+self.dataset_path+'/'+self.folder+'/'+self.filename+'"'
        if self.frame is not -1:
           string= string+ ' @'+ str(self.frame)
        string=string+' : '
        n_obj=len(self.rects)
        for rectangle in self.rects:
            string = string + rectangle.get_code_string()
            n_obj=n_obj-1
            if n_obj>0:
                string= string + ','
            else: string= string + ';'
        return string

    def get_rects_chall(self):
        """Get the string of the coordinates of all the rects of the image."""
        string='"'+self.dataset_path+'/'+self.folder+'/'+self.filename+'"'
        if self.frame is not -1:
           string= string+ ' @'+ str(self.frame)
        string=string+' : '
        n_obj=len(self.rects)
        for rectangle in self.rects:
            string = string + rectangle.get_chall_string()
            n_obj=n_obj-1
            if n_obj>0:
                string= string + ','
            else: string= string + ';'
        return string

    def get_info_string(self):
        """Get the string of the infos of the image."""
        return self.get_rects_string()

### Save frames array function to .idl

def saveVideoResults(filename, annotations):
    if not os.path.exists(filename):
        print "Created File: "+ filename
    file = open(filename, 'w')
    for annotation in annotations:
        frame = -1
        trackID=-1
        conf=0
        silhouette=-1
        xmin,ymin,xmax,ymax=0,0,0,0

        detections_array=[]

        if annotation.frame is not -1:
            frame=annotation.frame
        for rect in annotation.rects:
            if vid_classes.class_string_to_comp_code(rect.label) is not 'nothing':
                silhouette=rect.label
            if rect.trackID is not -1:
                trackID=rect.trackID
            conf = rect.true_confidence
            xmin,ymin,xmax,ymax = rect.x1,rect.y1,rect.x2 ,rect.y2
            file.write(str(frame)+' '+str(silhouette)+' '+str(trackID)+' '+str(conf)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax) + os.linesep)
    file.close()
