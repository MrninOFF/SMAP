#GUI
import tkinter as tk
import tkinter.ttk as ttk
import subprocess
import os as osCommand
import sched, time, threading
from datetime import datetime
#Camera
import io as cameraIO
import picamera
import cv2
import numpy
#Neuronka
import numpy
from typing import Tuple, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets, io
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import pandas as pd
from PIL import Image
#Muzika
import pygame

        
# Definice pomocne tridy pro neuronku
class CweFace(pl.LightningModule):

    def __init__(self, num_target_classes):
        print("Nacitam model")
        super().__init__()
        self.model = torch.jit.load("model_final_epoch_01.ts",map_location='cpu')
        self.acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

print(torch.__version__)
isRunning = 0
lastFaceEmotion = 100

#Neuronka
model = CweFace(4)
loader = transforms.Compose([
    transforms.Resize(244),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#Kamera
face_cascade = cv2.CascadeClassifier('/home/pi/project_folder/haarcascade_frontalface_default.xml')

#Hudba
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("sound.mp3")

#Klavesnice
keyWindow = tk.Tk()  # key window name
keyWindow.title('Keyboard blocker by face recongnition')  # title Name

# Size window size
keyWindow.geometry('1110x700')                    # normal size
keyWindow.maxsize(width=1110, height=700)         # maximum size
keyWindow.minsize(width= 1110 , height = 700)     # minimum size

keyWindow.configure(bg = 'white')    #  add background color

#button style
SUNKABLE_BUTTON_ON = 'SunkableButtonON.TButton'
SUNKABLE_BUTTON_OFF = 'SunkableButtonOFF.TButton'

style = ttk.Style()
style.configure(SUNKABLE_BUTTON_OFF, relief = 'tk.SUNKEN', foreground = 'black', background = 'gray')
style.configure(SUNKABLE_BUTTON_ON, relief = 'tk.RAISED', foreground = 'white', background = 'red')

#Hlavni program:

def pressOn(btn):
        btn.configure(command = lambda : pressOff(btn), style = SUNKABLE_BUTTON_OFF)
        
def pressOff(btn):
        btn.configure(command = lambda : pressOn(btn), style = SUNKABLE_BUTTON_ON)
        
def mainProgramStart():
    global isRunning
    isRunning = 1
    osCommand.system("xmodmap -pke > .Xmodmap")  #A
    #subprocess.call(['xmodmap','-e', 'keycode 10= '])
    #subprocess.call(['xmodmap','-e', 'keycode 20 = '])
    #subprocess.call(['xmodmap','-e', 'keycode 30 = '])
    #subprocess.call(['xmodmap','-pke'])
    #os.system("xmodmap .Xmodmap")
    threading.Timer(10, mainProgram).start()
def mainProgramStop():
    #Lagne celej system
    osCommand.system("xmodmap .Xmodmap")
    global isRunning
    isRunning = 0
    
def image_loader(image_name):
    """load image, tensor"""
    custom_image = Image.open(image_name).convert('RGB')
    custom_image = loader(custom_image).float()
    custom_image = custom_image.unsqueeze(0)
    return custom_image         
#MAIN CODE
def mainProgram():
    print("Start")
    if isRunning is 1:
        nextBreakTime = 10
        print("Take image")
        stream = cameraIO.BytesIO()
        with picamera.PiCamera() as camera:
            camera.resolution = (1920 , 1080 )
            camera.capture(stream, format='jpeg')
        buff = numpy.frombuffer(stream.getvalue(), dtype=numpy.uint8)
        image = cv2.imdecode(buff, 1)
        print("Find face in image")
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        size = 0
        print ("Found {}" + str(len(faces)) + " face(s)")
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                if((w*h) > size):
                    size = w*h
                    # - Ctverec kolem oblicejecv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),4)
                    img_cropped = image[y-4:y+h+4, x-4:x+w+4]
                    cv2.imwrite('result_gray_cut.jpg',img_cropped)
            print("Transform image")
            class_names = ['angry', 'happy', 'neutral', 'sad'] # Pro info
            imagePred = image_loader("result_gray_cut.jpg")
            #print(imagePred) #Kontrola
            outputs = model(imagePred)
            _, preds = torch.max(outputs, 1)
            print("Nalada je: ", class_names[preds])
            global lastFaceEmotion
            if (lastFaceEmotion is not preds):
                #Pokud se neznenil stav, nema smysl cokoli upravovat
                if (preds is 0):
                   print("Block keyboard")
                   subprocess.call(['xmodmap','-e', 'keycode 36= '])
                   subprocess.call(['xmodmap','-e', 'keycode 104= '])
                   nextBreakTime = 20
                if (preds is 1 or preds is 2):
                    print("Unblock keyboard")
                    pygame.mixer.music.stop()
                    subprocess.call(['xmodmap','-e', 'keycode 36= KP_Enter NoSymbol KP_Enter'])
                    subprocess.call(['xmodmap','-e', 'keycode 104= KP_Enter NoSymbol KP_Enter'])
                if (preds is 0 or preds is 3):
                    print("Play music")
                    pygame.mixer.music.play()
                    nextBreakTime = 15
            print("Take pause")
            lastFaceEmotion = preds
        else:
            print("Nenašel jsem obličej na fotografii")
            jmeno = "noFace" + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".jpg"
            print("noFace",datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),".jpg", sep='')
            cv2.imwrite(jmeno,image)
        threading.Timer(nextBreakTime, mainProgram).start()

#GUI
# First line
Q = ttk.Button(keyWindow,text = 'Q',width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Q))
Q.grid(row = 1 , column = 0, ipadx = 6 , ipady = 10)

W = ttk.Button(keyWindow,  text = 'W' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(W))
W.grid(row = 1 , column = 1, ipadx = 6 , ipady = 10)

E = ttk.Button(keyWindow,  text = 'E' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(E))
E.grid(row = 1 , column = 2, ipadx = 6 , ipady = 10)

R = ttk.Button(keyWindow,  text = 'R' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(R))
R.grid(row = 1 , column = 3, ipadx = 6 , ipady = 10)

T = ttk.Button(keyWindow,  text = 'T' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(T))
T.grid(row = 1 , column = 4, ipadx = 6 , ipady = 10)

Y = ttk.Button(keyWindow,  text = 'Y' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Y))
Y.grid(row = 1 , column = 5, ipadx = 6 , ipady = 10)

U = ttk.Button(keyWindow,  text = 'U' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(U))
U.grid(row = 1 , column = 6, ipadx = 6 , ipady = 10)

I = ttk.Button(keyWindow,  text = 'I' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(I))
I.grid(row = 1 , column = 7, ipadx = 6 , ipady = 10)

O = ttk.Button(keyWindow,  text = 'O' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(O))
O.grid(row = 1 , column = 8, ipadx = 6 , ipady = 10)

P = ttk.Button(keyWindow,  text = 'P' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(P))
P.grid(row = 1 , column = 9, ipadx = 6 , ipady = 10)

cur = ttk.Button(keyWindow,  text = '{' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(cur))
cur.grid(row = 1 , column = 10 , ipadx = 6 , ipady = 10)

cur_c = ttk.Button(keyWindow,  text = '}' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(cur_c))
cur_c.grid(row = 1 , column = 11, ipadx = 6 , ipady = 10)

back_slash = ttk.Button(keyWindow,  text = '\\' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(back_slash))
back_slash.grid(row = 1 , column = 12, ipadx = 6 , ipady = 10)


clear = ttk.Button(keyWindow,  text = 'Clear' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(clear))
clear.grid(row = 1 , column = 13, ipadx = 20 , ipady = 10)

# Second Line Button



A = ttk.Button(keyWindow,  text = 'A' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(A))
A.grid(row = 2 , column = 0, ipadx = 6 , ipady = 10)



S = ttk.Button(keyWindow,  text = 'S' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(S))
S.grid(row = 2 , column = 1, ipadx = 6 , ipady = 10)

D = ttk.Button(keyWindow,  text = 'D' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(D))
D.grid(row = 2 , column = 2, ipadx = 6 , ipady = 10)

F = ttk.Button(keyWindow,  text = 'F' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(F))
F.grid(row = 2 , column = 3, ipadx = 6 , ipady = 10)


G = ttk.Button(keyWindow,  text = 'G' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(G))
G.grid(row = 2 , column = 4, ipadx = 6 , ipady = 10)


H = ttk.Button(keyWindow,  text = 'H' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(H))
H.grid(row = 2 , column = 5, ipadx = 6 , ipady = 10)


J = ttk.Button(keyWindow,  text = 'J' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(J))
J.grid(row = 2 , column = 6, ipadx = 6 , ipady = 10)


K = ttk.Button(keyWindow,  text = 'K' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(K))
K.grid(row = 2 , column = 7, ipadx = 6 , ipady = 10)

L = ttk.Button(keyWindow,  text = 'L' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(L))
L.grid(row = 2 , column = 8, ipadx = 6 , ipady = 10)


semi_co = ttk.Button(keyWindow,  text = ';' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(semi_co))
semi_co.grid(row = 2 , column = 9, ipadx = 6 , ipady = 10)


d_colon = ttk.Button(keyWindow,  text = '"' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(d_colon))
d_colon.grid(row = 2 , column = 10, ipadx = 6 , ipady = 10)


enter = ttk.Button(keyWindow,  text = 'Enter' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(enter))
enter.grid(row = 2 , column = 13, ipadx = 20 , ipady = 10)

# third line Button

Z = ttk.Button(keyWindow,  text = 'Z' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Z))
Z.grid(row = 3 , column = 0, ipadx = 6 , ipady = 10)


X = ttk.Button(keyWindow,  text = 'X' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(X))
X.grid(row = 3 , column = 1, ipadx = 6 , ipady = 10)


C = ttk.Button(keyWindow,  text = 'C' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(C))
C.grid(row = 3 , column = 2, ipadx = 6 , ipady = 10)


V = ttk.Button(keyWindow,  text = 'V' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(V))
V.grid(row = 3 , column = 3, ipadx = 6 , ipady = 10)

B = ttk.Button(keyWindow, text= 'B' , width = 6 , style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(B))
B.grid(row = 3 , column = 4 , ipadx = 6 ,ipady = 10)


N = ttk.Button(keyWindow,  text = 'N' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(N))
N.grid(row = 3 , column = 5, ipadx = 6 , ipady = 10)


M = ttk.Button(keyWindow,  text = 'M' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(M))
M.grid(row = 3 , column = 6, ipadx = 6 , ipady = 10)


left = ttk.Button(keyWindow,  text = '<' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(left))
left.grid(row = 3 , column = 7, ipadx = 6 , ipady = 10)


right = ttk.Button(keyWindow,  text = '>' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(right))
right.grid(row = 3 , column = 8, ipadx = 6 , ipady = 10)


slas = ttk.Button(keyWindow,  text = '/' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(slas))
slas.grid(row = 3 , column = 9, ipadx = 6 , ipady = 10)


q_mark = ttk.Button(keyWindow,  text = '?' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(q_mark))
q_mark.grid(row = 3 , column = 10, ipadx = 6 , ipady = 10)


coma = ttk.Button(keyWindow,  text = ',' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(coma))
coma.grid(row = 3 , column = 11, ipadx = 6 , ipady = 10)

dot = ttk.Button(keyWindow,  text = '.' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(dot))
dot.grid(row = 3 , column = 12, ipadx = 6 , ipady = 10)

shift = ttk.Button(keyWindow,  text = 'Shift' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(shift))
shift.grid(row = 3 , column = 13, ipadx = 20 , ipady = 10)

#Fourth Line Button


ctrl = ttk.Button(keyWindow,  text = 'Ctrl' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(ctrl))
ctrl.grid(row = 4 , column = 0, ipadx = 6 , ipady = 10)


Fn = ttk.Button(keyWindow,  text = 'Fn' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Fn))
Fn.grid(row = 4 , column = 1, ipadx = 6 , ipady = 10)


window = ttk.Button(keyWindow,  text = 'Window' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(window))
window.grid(row = 4 , column = 2 , ipadx = 6 , ipady = 10)

Alt = ttk.Button(keyWindow,  text = 'Alt' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Alt))
Alt.grid(row = 4 , column = 3 , ipadx = 6 , ipady = 10)

space = ttk.Button(keyWindow,  text = 'Space' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(space))
space.grid(row = 4 , columnspan = 14 , ipadx = 160 , ipady = 10)

Alt_gr = ttk.Button(keyWindow,  text = 'Alt Gr' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(Alt_gr))
Alt_gr.grid(row = 4 , column = 10 , ipadx = 6 , ipady = 10)

open_b = ttk.Button(keyWindow,  text = '(' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(open_b))
open_b.grid(row = 4 , column = 11 , ipadx = 6 , ipady = 10)

close_b = ttk.Button(keyWindow,  text = ')' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(close_b))
close_b.grid(row = 4 , column = 12 , ipadx = 6 , ipady = 10)


tap = ttk.Button(keyWindow,  text = 'Tab' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : pressOff(tap))
tap.grid(row = 4 , column = 13 , ipadx = 20 , ipady = 10)

start = ttk.Button(keyWindow,  text = 'START' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : mainProgramStart())
start.grid(row = 5 , column = 1 , ipadx = 6 , ipady = 10)

start = ttk.Button(keyWindow,  text = 'STOP' , width = 6, style = SUNKABLE_BUTTON_OFF ,command = lambda : mainProgramStop())
start.grid(row = 5 , column = 2 , ipadx = 6 , ipady = 10)

keyWindow.mainloop()