import os

def checkPath(path):
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print filename

def get_Files_List(path):
    files_list=[]
    for path, subdirs,files in os.walk(path):
        for filename in files:
            files_list.append(os.path.join(path, filename))
    return files_list
