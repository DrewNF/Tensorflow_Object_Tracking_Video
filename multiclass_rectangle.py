import copy 
import utils_video

#####################################################################################################################################################################
########################## CLASS AND FUNCTIONS DEFINED TO USE WITH TENSORBOX (MAINLY TO PARSE & WRITE .IDL FILE AND MANAGE RESULTS) #################################
#####################################################################################################################################################################


class Rectangle_Multiclass(object):



    def __init__(self):
        # Initialization Function
        # BBox Parameters

        self.cx = -1
        self.cy = -1
        self.width = -1
        self.height = -1
        self.true_confidence = -1
        self.x1 = -1
        self.x2 = -1
        self.y1 = -1
        self.y2 = -1      

        # Track Parameter
        
        self.trackID=-1

        # Label Parameters

        self.label_confidence = -1
        self.label= 'Not Set'
        self.label_chall='Not Set'
        self.label_code= 'Not Set'

    ### Safe Loading Values Functions

    def load_labeled_rect(self, trackID, rect_conf, label_conf, x1, x2, y1, y2, label, label_chall, code):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.label=label
        self.label_code=code
        self.label_chall=label_chall

        self.trackID=trackID

        self.label_confidence=label_conf
        self.true_confidence=rect_conf

    def load_BBox(self, x1, x2, y1, y2, label, label_chall, code):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.label=label
        self.label_code=code
        self.label_chall=label_chall

    def set_unlabeled_rect(self, cx, cy, width, height, confidence):
        # Set unlabeled rect info to be processed forward
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.true_confidence = confidence
        self.x1 = self.cx - self.width/2.
        self.x2 = self.cx + self.width/2.
        self.y1 = self.cy - self.height/2.
        self.y2 = self.cy + self.height/2.

    def add_delta(self, dx1, dx2, dy1, dy2):
        # Set unlabeled rect info to be processed forward
        self.x1 = self.x1+dx1 
        self.x2 = self.x2+dx2
        self.y1 = self.y1+dy1
        self.y2 = self.y2+dy2
        self.cx = (self.x1 + self.x2)/2.
        self.cy = (self.y1 + self.y2)/2.
        self.width = max(self.x1,self.x2) - min(self.x1,self.x2)
        self.height = max(self.y1,self.y2) - min(self.y1,self.y2)

    def set_rect_coordinates(self, x1, x2, y1, y2):
        # Set rect coordinates info to be processed forward

        self.x1 = x1 
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cx = (self.x1 + self.x2)/2.
        self.cy = (self.y1 + self.y2)/2.
        self.width = max(self.x1,self.x2) - min(self.x1,self.x2)
        self.height = max(self.y1,self.y2) - min(self.y1,self.y2)

    def load_label(self, trackID, label_conf, label, label_chall, code):

        self.label=label
        self.label_code=code
        self.label_chall=label_chall
        self.trackID=trackID
        self.label_confidence=label_conf

    def load_trackID(self, trackID):

        self.trackID=trackID

    def set_label(self, label_conf, label, label_chall, code):

        self.label=label
        self.label_code=code
        self.label_chall=label_chall
        self.label_confidence=label_conf

    def check_rects_motion(self,filename, rect, dx1, dx2, dy1,dy2, error=1.2, attenuation=1.1):
        ## Rect is considered passed befor through add_delta
        if((self.x1-rect.x1)>dx1*error)| ((self.y1-rect.y1)>dy1*error)|((self.x2-rect.x2)>dx2*error)|((self.y2-rect.y2)>dy2*error):
            utils_video.draw_rectangle(filename,(self.x1, self.y1,self.x2, self.y2))
            delta_cx=self.cx-rect.cx
            delta_cy=self.cy-rect.cy
            self.x1 =rect.x1 + delta_cx
            self.y1 =rect.y1 + delta_cy
            self.x2 =rect.x2 + delta_cx
            self.y2 =rect.y2 + delta_cy
            self.cx = (self.x1 + self.x2)/2.
            self.cy = (self.y1 + self.y2)/2.
            self.width = max(self.x1,self.x2) - min(self.x1,self.x2)
            self.height = max(self.y1,self.y2) - min(self.y1,self.y2)

    ### Safe Duplicate functions 

    def duplicate(self):
        new_rect=Rectangle_Multiclass()
        new_rect.cx = copy.copy(self.cx)
        new_rect.cy = copy.copy(self.cy)
        new_rect.width = copy.copy(self.width)
        new_rect.height = copy.copy(self.height)
        new_rect.true_confidence = copy.copy(self.true_confidence)
        new_rect.label_confidence = copy.copy(self.label_confidence)
        new_rect.label= copy.copy(self.label)
        new_rect.trackID=copy.copy(self.trackID)
        new_rect.x1 = copy.copy(self.x1)
        new_rect.x2 = copy.copy(self.x2)
        new_rect.y1 = copy.copy(self.y1)
        new_rect.y2 = copy.copy(self.y2)
        return new_rect

    ### Computation Functions

    def overlaps(self, other):
        if abs(self.cx - other.cx) > (self.width + other.width) / 1.5:
            return False
        elif abs(self.cy - other.cy) > (self.height + other.height) / 2.0:
            return False
        else:
            return True
    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))
    def intersection(self, other):
        left = max(self.cx - self.width/2., other.cx - other.width/2.)
        right = min(self.cx + self.width/2., other.cx + other.width/2.)
        width = max(right - left, 0)
        top = max(self.cy - self.height/2., other.cy - other.height/2.)
        bottom = min(self.cy + self.height/2., other.cy + other.height/2.)
        height = max(bottom - top, 0)
        return width * height
    def area(self):
        return self.height * self.width
    def union(self, other):
        return self.area() + other.area() - self.intersection(other)
    def iou(self, other):
        return self.intersection(other) / self.union(other)
    def __eq__(self, other):
        return (self.cx == other.cx and 
            self.cy == other.cy and
            self.width == other.width and
            self.height == other.height and
            self.confidence == other.confidence and
            self.label_confidence == other.label_confidence and self.label == other.label and self.trackID == other.trackID)

    ### Functions for parse the xml into the .idl file to train TENSORBOX

    def get_label_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label is not 'Not Set':
            string=self.label +' '
        return string


    def get_code_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label_code is not -1:
            string=self.label_code +' '
        return string


    def get_chall_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label_chall is not 'Not Set':
            string=str(self.label_chall) +' '
        return string
    
    def get_coord_string(self):
        """Get the string of the coordinates of the rect."""
        string='('+str(self.x1)+','+str(self.y1)+','+str(self.x2)+','+str(self.y2)+')'
        return string

    def get_rect_string(self):
        """Get the string of the infos of the rect."""
        string='('+str(self.x1)+','+str(self.y1)+','+str(self.x2)+','+str(self.y2)+')'
        if self.label_chall is not 'Not Set':
            string=string+' /' + str(self.label_chall) 
        return string


def duplicate_rects(rects):

    new_rects=[]
    for rect in rects:
        new_rect=Rectangle_Multiclass()
        new_rect.cx = copy.copy(rect.cx)
        new_rect.cy = copy.copy(rect.cy)
        new_rect.width = copy.copy(rect.width)
        new_rect.height = copy.copy(rect.height)
        new_rect.true_confidence = copy.copy(rect.true_confidence)
        new_rect.label_confidence = copy.copy(rect.label_confidence)
        new_rect.label= copy.copy(rect.label)
        new_rect.trackID=copy.copy(rect.trackID)
        new_rect.x1 = copy.copy(rect.x1)
        new_rect.x2 = copy.copy(rect.x2)
        new_rect.y1 = copy.copy(rect.y1)
        new_rect.y2 = copy.copy(rect.y2)
        new_rects.append(new_rect)
    return new_rects

def pop_max_iou(rects, rect):
    max_iou=None
    max_id=0
    rect_id=0
    for rectangle in rects:
        if max_iou is None:
            max_iou=rect.iou(rectangle)
            max_id=rect_id
        if rect.iou(rectangle)>max_iou:
            max_iou=rect.iou(rectangle)
            max_id=rect_id
        rect_id=rect_id+1
    if len(rects)>max_id:
        new_rect=rects[max_id].duplicate()
        rects.pop(max_id)
        return new_rect
    else: return None 

def pop_max_overlap(rects, rect):
    max_overlap=None
    max_id=0
    rect_id=0
    for rectangle in rects:
        if max_overlap is None:
            max_overlap=rect.overlaps(rectangle)
            max_id=rect_id
        if rect.iou(rectangle)>max_overlap:
            max_overlap=rect.overlaps(rectangle)
            max_id=rect_id
        rect_id=rect_id+1
    if len(rects)>max_id:
        new_rect=rects[max_id].duplicate()
        rects.pop(max_id)
        return new_rect
    else: return None 