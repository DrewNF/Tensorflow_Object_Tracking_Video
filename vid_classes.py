####DEFINE COLOR SWITCH

def code_to_class_string(argument):
    switcher = {
                    'n02691156': "airplane",
                    'n02419796': "antelope",
                    'n02131653': "bear",
                    'n02834778': "bicycle",
                    'n01503061': "bird",
                    'n02924116': "bus",
                    'n02958343': "car",
                    'n02402425': "cattle",
                    'n02084071': "dog",
                    'n02121808': "domestic_cat",
                    'n02503517': "elephant",
                    'n02118333': "fox",
                    'n02510455': "giant_panda",
                    'n02342885': "hamster",
                    'n02374451': "horse",
                    'n02129165': "lion",
                    'n01674464': "lizard",
                    'n02484322': "monkey",
                    'n03790512': "motorcycle",
                    'n02324045': "rabbit",
                    'n02509815': "red_panda",
                    'n02411705': "sheep",
                    'n01726692': "snake",
                    'n02355227': "squirrel",
                    'n02129604': "tiger",
                    'n04468005': "train",
                    'n01662784': "turtle",
                    'n04530566': "watercraft",
                    'n02062744': "whale",
                    'n02391049': "zebra"            }
    return switcher.get(argument, "nothing")

def code_to_code_chall(argument):
    switcher = {
                    'n02691156': 1,
                    'n02419796': 2,
                    'n02131653': 3,
                    'n02834778': 4,
                    'n01503061': 5,
                    'n02924116': 6,
                    'n02958343': 7,
                    'n02402425': 8,
                    'n02084071': 9,
                    'n02121808': 10,
                    'n02503517': 11,
                    'n02118333': 12,
                    'n02510455': 13,
                    'n02342885': 14,
                    'n02374451': 15,
                    'n02129165': 16,
                    'n01674464': 17,
                    'n02484322': 18,
                    'n03790512': 19,
                    'n02324045': 20,
                    'n02509815': 21,
                    'n02411705': 22,
                    'n01726692': 23,
                    'n02355227': 24,
                    'n02129604': 25,
                    'n04468005': 26,
                    'n01662784': 27,
                    'n04530566': 28,
                    'n02062744': 29,
                    'n02391049': 30            }
    return switcher.get(argument, "nothing")

def class_string_to_comp_code(argument):
    switcher = {
                    'airplane': 1,
                    'antelope': 2,
                    'bear': 3,
                    'bicycle': 4,
                    'bird': 5,
                    'bus': 6,
                    'car': 7,
                    'cattle': 8,
                    'dog': 9,
                    'domestic_cat': 10,
                    'elephant': 11,
                    'fox': 12,
                    'giant_panda': 13,
                    'hamster': 14,
                    'horse': 15,
                    'lion': 16,
                    'lizard': 17,
                    'monkey': 18,
                    'motorcycle': 19,
                    'rabbit': 20,
                    'red_panda': 21,
                    'sheep': 22,
                    'snake': 23,
                    'squirrel': 24,
                    'tiger': 25,
                    'train': 26,
                    'turtle': 27,
                    'watercraft': 28,
                    'whale': 29,
                    'zebra': 30                 
               }
    return switcher.get(argument, None)

def code_comp_to_class(argument):
    switcher = {
                    1:'airplane',
                    2:'antelope',
                    3:'bear',
                    4:'bicycle',
                    5:'bird',
                    6:'bus',
                    7:'car',
                    8:'cattle',
                    9:'dog',
                    10:'domestic_cat',
                    11:'elephant',
                    12:'fox',
                    13:'giant_panda',
                    14:'hamster',
                    15:'horse',
                    16:'lion',
                    17:'lizard',
                    18:'monkey',
                    19:'motorcycle',
                    20:'rabbit',
                    21:'red_panda',
                    22:'sheep',
                    23:'snake',
                    24:'squirrel',
                    25:'tiger',
                    26:'train',
                    27:'turtle',
                    28:'watercraft',
                    29:'whale',
                    30:'zebra'                 }
    return switcher.get(argument, "nothing")



### Color Switching 

def name_string_to_color(argument):
    switcher = {
                    'airplane': 'black' ,
                    'antelope': 'white',
                    'bear': 'red',
                    'bicycle': 'lime' ,
                    'bird': 'blue',
                    'bus': 'yellow',
                    'car': 'cyan',
                    'cattle':  'magenta',
                    'dog': 'silver',
                    'domestic_cat': 'gray' ,
                    'elephant':'maroon' ,
                    'fox':'olive' ,
                    'giant_panda':'green' ,
                    'hamster':'purple' ,
                    'horse':'teal' ,
                    'lion':'navy' ,
                    'lizard':'pale violet red' ,
                    'monkey':'deep pink' ,
                    'motorcycle':'aqua marine' ,
                    'rabbit':'powder blue' ,
                    'red_panda':'spring green' ,
                    'sheep':'sea green' ,
                    'snake':'forest green' ,
                    'squirrel':'orange red' ,
                    'tiger':'dark orange' ,
                    'train':'orange' ,
                    'turtle':'dark golden rod' ,
                    'watercraft':'golden rod' ,
                    'whale':'dark red' ,
                    'zebra':'light coral'                  }
    return switcher.get(argument, "nothing")

def code_to_color(argument):
    switcher = {
                    1:(0,0,0),
                    2:(255,255,255),
                    3:(255,0,0),
                    4:(0,255,0),
                    5:(0,0,255),
                    6:(255,255,0),
                    7:(0,255,255),
                    8:(255,0,255),
                    9:(192,192,192),
                    10:(128,128,128),
                    11:(128,0,0),
                    12:(128,128,0),
                    13:(0,128,0),
                    14:(128,0,128),
                    15:(0,128,128),
                    16:(0,0,128),
                    17:(219,112,147),
                    18:(255,20,147),
                    19:(127,255,212),
                    20:(176,224,230),
                    21:(0,255,127),
                    22:(46,139,87),
                    23:(34,139,34),
                    24:(255,69,0),
                    25:(255,140,0),
                    26:(255,165,0),
                    27:(184,134,11),
                    28:(218,165,32),
                    29:(139,0,0),
                    30:(240,128,128) 
                      }
    return switcher.get(argument,(0,0,0) )

def label_to_color(argument):
    switcher = {
                    'n02691156':(0,0,0),
                    'n02419796':(255,255,255),
                    'n02131653':(255,0,0),
                    'n02834778':(0,255,0),
                    'n01503061':(0,0,255),
                    'n02924116':(255,255,0),
                    'n02958343':(0,255,255),
                    'n02402425':(255,0,255),
                    'n02084071':(192,192,192),
                    'n02121808':(128,128,128),
                    'n02503517':(128,0,0),
                    'n02118333':(128,128,0),
                    'n02510455':(0,128,0),
                    'n02342885':(128,0,128),
                    'n02374451':(0,128,128),
                    'n02129165':(0,0,128),
                    'n01674464':(219,112,147),
                    'n02484322':(255,20,147),
                    'n03790512':(127,255,212),
                    'n02324045':(176,224,230),
                    'n02509815':(0,255,127),
                    'n02411705':(46,139,87),
                    'n01726692':(34,139,34),
                    'n02355227':(255,69,0),
                    'n02129604':(255,140,0),
                    'n04468005':(255,165,0),
                    'n01662784':(184,134,11),
                    'n04530566':(218,165,32),
                    'n02062744':(139,0,0),
                    'n02391049':(240,128,128) 
                      }
    return switcher.get(argument,(0,0,0) )


 ####### COLOR LEGEND #######
 #		Black 	#000000 	(0,0,0)
 #  	White 	#FFFFFF 	(255,255,255)
 #  	Red 	#FF0000 	(255,0,0)
 #  	Lime 	#00FF00 	(0,255,0)
 #  	Blue 	#0000FF 	(0,0,255)
 #  	Yellow 	#FFFF00 	(255,255,0)
 #  	Cyan / Aqua 	#00FFFF 	(0,255,255)
 #  	Magenta / Fuchsia 	#FF00FF 	(255,0,255)
 #  	Silver 	#C0C0C0 	(192,192,192)
 #  	Gray 	#808080 	(128,128,128)
 #  	Maroon 	#800000 	(128,0,0)
 #  	Olive 	#808000 	(128,128,0)
 #  	Green 	#008000 	(0,128,0)
 #  	Purple 	#800080 	(128,0,128)
 #  	Teal 	#008080 	(0,128,128)
 #  	Navy 	#000080 	(0,0,128)
 #  	pale violet red 	#DB7093 	(219,112,147)
 #  	deep pink 	#FF1493 	(255,20,147)
 #  	aqua marine 	#7FFFD4 	(127,255,212)
 #  	powder blue 	#B0E0E6 	(176,224,230)
 #  	spring green 	#00FF7F 	(0,255,127)
 #  	sea green 	#2E8B57 	(46,139,87)
 #  	forest green 	#228B22 	(34,139,34)
 #  	lime 	#00FF00 	(0,255,0)
 #  	orange red 	#FF4500 	(255,69,0)
 #  	dark orange 	#FF8C00 	(255,140,0)
 #  	orange 	#FFA500 	(255,165,0)
 #  	dark golden rod 	#B8860B 	(184,134,11)
 #  	golden rod 	#DAA520 	(218,165,32)
 #  	dark red 	#8B0000 	(139,0,0)
 #  	light coral 	#F08080 	(240,128,128)

class Classes_List(object):
        
     class_name_string_list= ['airplane','antelope','bear','bicycle','bird','bus','car','cattle','dog','domestic_cat','elephant','fox','giant_panda','hamster','horse','lion','lizard','monkey','motorcycle','rabbit','red_panda','sheep','snake','squirrel','tiger','train','turtle','watercraft','whale','zebra']

     class_code_string_list= ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']

     colors_string_list=['black' ,'white','red','lime' ,'blue','yellow','cyan', 'magenta','silver', 'gray' ,'maroon' ,'olive' ,'green' ,'purple' ,'teal' ,'navy' ,'pale violet red' ,'deep pink' ,'aqua marine' ,'powder blue' ,'spring green' ,'sea green' ,'forest green' ,'orange red' ,'dark orange' ,'orange' ,'dark golden rod' ,'golden rod' ,'dark red' ,'light coral'  ]

     colors_code_list=[(0,0,0),(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(192,192,192),(128,128,128),(128,0,0),(128,128,0),(0,128,0),(128,0,128),(0,128,128),(0,0,128),(219,112,147),(255,20,147),(127,255,212),(176,224,230),(0,255,127),(46,139,87),(34,139,34),(255,69,0),(255,140,0),(255,165,0),(184,134,11),(218,165,32),(139,0,0),(240,128,128)]
