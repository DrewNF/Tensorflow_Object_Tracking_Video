#### Bounding Boxes Informations Class
class BB_Rectangle(object):

    def __init__(self):
        """Return a rect object whose coords are *0* and infos none ."""
        self.x1 = 0.0 #xmin x1 letf
        self.y1 = 0.0 #ymin y1 bottom
        self.x2 = 0.0 #xmax x2 right
        self.y2 = 0.0 #ymax y2 top
        self.label=None
        self.label_code=None
        self.label_chall=None

    def BB_rect(self, x1, x2, y1, y2, label, label_chall, code):
    	self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label=label
        self.label_code=code
        self.label_chall=label_chall
        return self 


    def get_label_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label is not None:
            string=self.label +' '
        return string


    def get_code_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label_code is not None:
            string=self.label_code +' '
        return string


    def get_chall_string(self):
        """Get the string of the label of the rect."""
        string=""
        if self.label_chall is not None:
            string=str(self.label_chall) +' '
        return string
    
    def get_coord_string(self):
        """Get the string of the coordinates of the rect."""
        string='('+str(self.x1)+','+str(self.y1)+','+str(self.x2)+','+str(self.y2)+')'
        return string

    def get_rect_string(self):
        """Get the string of the infos of the rect."""
        string='('+str(self.x1)+','+str(self.y1)+','+str(self.x2)+','+str(self.y2)+')'
        if self.label_chall is not None:
            string=string+' /' + str(self.label_chall) 
        return string

#### Picture Informations Class

class Picture_Info(object):
    
    def __init__(self):
        """Return a picture_info object whose rects are *None*."""
        self.rects = []
        self.folder=''
        self.filename=''
        self.dataset_path=''
        self.default_path=''
        self.width=-1
        self.height=-1
        self.frame=-1
    
    def append_rect(self, rectangle):
        """Adding rect to the picture_info."""
        rect= BB_Rectangle().BB_rect( rectangle.x1, rectangle.x2, rectangle.y1, rectangle.y2, rectangle.label, rectangle.label_chall, rectangle.label_code)
        index= len(self.rects)
        self.rects.insert(index, rect)

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


