import copy 

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
        self.label_code=-1

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

    def load_label(self, trackID, label_conf, label, label_chall, code):


        self.label=label
        self.label_code=code
        self.label_chall=label_chall

        self.trackID=trackID

        self.label_confidence=label_conf

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
