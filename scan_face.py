import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import PIL
import mysql.connector
import numpy as np

class face_recognition:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1530x1080+0+0")

        img = Image.open("images\scanface.jpg")
        img = img.resize((1920, 900), Image.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(img)

        fir_label = tk.Label(self.root, image=self.photoimage)
        fir_label.place(x=0, y=50, width=1550, height=750)

        title_label = tk.Label(self.root, text="FACE RECOGNITION SYSTEM", font=(
            "Arial", 35, "bold"), bg="black", fg="white")
        title_label.place(x=0, y=0, width=1600, height=50)

        img1 = Image.open(
            r"images/scanbtn.jpg")
        img1 = img1.resize((250, 250), PIL.Image.LANCZOS)
        self.photoimage1 = ImageTk.PhotoImage(img1)

        b1 = Button(self.root, image=self.photoimage1,
                    highlightbackground="white", highlightthickness=10,command=self.face_rec,cursor="hand2")
        b1.place(x=1300, y=280, height=209, width=190)

        btn1_label = Label(self.root,text="scan", font=(
            "Arial", 15, "bold"), bg="white", fg="red")
        btn1_label.place(x=1300, y=450, width=190, height=40)

        label = Label(self.root,text="Press Scan BUtton", font=(
            "Arial", 15, "bold"), bg="white", fg="red")
        label.place(x=150, y=150, width=600, height=500)



    def face_rec(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(
                gray_image, scaleFactor, minNeighbours)

            co_ord = []

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, predict = clf.predict(gray_image[y:y+h, x:x+w])
                confidence = int((100*(1-predict/300)))

                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="admin",
                    database="face_recognition"
                )
                my_cursor = conn.cursor()

                my_cursor.execute("select Name from student where ID="+str(id))
                n = my_cursor.fetchone()
                n = str(n)

                my_cursor.execute("select Roll_No from student where ID="+str(id))
                r = my_cursor.fetchone()
                #r = "+".join(r)

                my_cursor.execute(
                    "select Department from student where ID="+str(id))
                d = my_cursor.fetchone()
                #d = "+".join(d)

                my_cursor.execute("select ID from student where ID="+str(id))
                si = my_cursor.fetchone()
                #si = "+".join(si)

                if confidence > 80:
                    cv2.putText(
                        img, f"ID:{si}", (x, y-85), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(
                        img, f"Roll:{r}", (x, y-55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(
                        img, f"Name:{n}", (x, y-30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(
                        img, f"Department:{d}", (x, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

                else:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown Face", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                co_ord = [x, y, w, h]

            return co_ord

        def recognize(img, clf, faceCascade):
            co_ord = draw_boundary(img, faceCascade, 1.1,
                                   10, (255, 255, 255), "Face", clf)
            return img

        faceCascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, img = video_capture.read()
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome to face Recognition", img)

            if cv2.waitKey(1) == 13:
                break
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    obj = face_recognition(root)
    root.mainloop()