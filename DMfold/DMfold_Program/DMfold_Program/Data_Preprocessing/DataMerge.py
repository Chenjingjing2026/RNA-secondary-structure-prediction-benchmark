import os
'''
This code is used to obtain all file names under a path and merge the absolute path of a file. 
'''
def EachFile(filepath):
    pathDir = os.listdir(filepath)
    return pathDir
def FileNmae(filepath, file):
    return filepath + '/' + file