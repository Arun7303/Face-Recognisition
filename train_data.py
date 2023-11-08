from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import PIL
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np

class train:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1530x1080+0+0")



        img = Image.open(
            r"D:\Vishwakarma University\SY\ISF\FaceRecognisation\images\traindata.jpg")
        img = img.resize((1900, 930), PIL.Image.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(img)

        fir_label = Label(self.root, image=self.photoimage)
        fir_label.place(x=0, y=0, width=1580, height=800)

        title_label = Label(self.root, text="TRAIN DATA", font=(
            "Arial", 35, "bold"), bg="black", fg="white")
        title_label.place(x=0, y=30, width=1600, height=50)

        btn_train = Image.open(
            r"D:\Vishwakarma University\SY\ISF\FaceRecognisation\images\traaindatabutton.jpg")
        btn_train = btn_train.resize((180, 180), PIL.Image.LANCZOS)
        self.photoimage1 = ImageTk.PhotoImage(btn_train)

        b1 = Button(self.root, image=self.photoimage1,
                    highlightbackground="black",command=self.train_classifier, highlightthickness=5,cursor="hand2")
        b1.place(x=695, y=400, height=150, width=190)

        b1_label = Label(self.root, text="TRAIN", font=(
            "Arial", 15, "bold"), bg="white", fg="black")
        b1_label.place(x=695, y=555, width=190, height=40)

        txt1 = Label(self.root, text="Face Database", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt1.place(x=200, y=288, width=190, height=40)

        txt2 = Label(self.root, text="Face Classification", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt2.place(x=1215, y=288, width=190, height=40)

        txt3 = Label(self.root, text="Facial Extraction", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt3.place(x=250, y=480, width=190, height=40)

        txt3 = Label(self.root, text="Facial Features", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt3.place(x=1150, y=480, width=190, height=40)
        
        txt4 = Label(self.root, text="Train Data", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt4.place(x=400, y=670, width=190, height=40)

        txt5 = Label(self.root, text="Result", font=(
            "Arial", 15, "bold"), bg="black", fg="grey")
        txt5.place(x=1000, y=670, width=190, height=40)



    def train_classifier(self):
        data_dir=("data")
        path=[os.path.join(data_dir,file)for file in os.listdir(data_dir)]

        
        faces=[]
        ids=[]
        

        for image in path:
            img=Image.open(image).convert('L')  #grey scale image
            imageNp=np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])


            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1)==13
        
        ids=np.array(ids)

        # train the classifier and save

        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","Training datasets completed!!",parent=self.root)

if __name__ == "__main__":
    root = Tk()
    obj = train(root)
    root.mainloop()