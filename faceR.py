from tkinter import *
import numpy as np 
import face_recognition as fr 
from PIL import Image, ImageTk
import cv2

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("700x700")

# Create a Label to capture the Video frames
label =Label(win)
cam= cv2.VideoCapture(0)
label.grid(column=0,row=0)
#training stuff
my_image = fr.load_image_file("myself.jpg")
mam_image = fr.load_image_file("madam1.jpg")

my_face_encoding = fr.face_encodings(my_image)[0]
mam_face_encoding = fr.face_encodings(mam_image)[0]


known_face_encondings = [my_face_encoding,mam_face_encoding]
known_face_names = ["Bargav","Correspondent mam"]

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cam.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   label.after(20, show_frames)

def capture():
    frame = cam.read()
    rgb_frame = frame[::-1]
    
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


cpt = Button(win , text="Capture" , command=capture, )
cpt.grid(column=0,row=2)

show_frames()
win.mainloop()