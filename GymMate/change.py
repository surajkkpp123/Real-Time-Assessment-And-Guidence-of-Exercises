import os
import shutil

path1=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\output.mp4"
path2=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\gymmate\static\output.mp4"
os.rename(path2,path1)
shutil.move(path2,path1)
os.replace(path2,path1)