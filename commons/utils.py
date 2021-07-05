import os
import datetime
import inspect

def find_all_files(directory,extention="csv"):
    filelist=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file[-3:] == extention:
                filelist.append(os.path.join(root, file))
    return filelist

class logger:
    _path=""
    def __init__(self,filepath):
       self._path=filepath

    def write(self,message):
        #Get Stackframe
        stack_frame = inspect.stack()[1]
        frame = stack_frame[0]
        info = inspect.getframeinfo(frame)
        #Create LoggingLine
        linetxt = ""
        linetxt += datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S ")
        linetxt += (info.filename + " ")
        linetxt += (str(info.lineno) + " ")
        linetxt += message
        linetxt += "\n"

        #Write
        with open(self._path, "a") as f:
            f.write(linetxt)

