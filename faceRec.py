import numpy as np
import face_recognition as fr
import cv2
import pyttsx3

video_capture = cv2.VideoCapture(0)

myImage = fr.load_image_file("myself.jpg")
mamImage = fr.load_image_file("madam1.jpg")
gregorySirPic = fr.load_image_file("Gregory-PT.jpg")
beenaMamPic = fr.load_image_file("Beena Stephen - X B.JPG")
deviMamPic = fr.load_image_file("Devi T N.JPG")
yogaSir= fr.load_image_file("jayaraman.JPG")
csMamPic = fr.load_image_file("Manjula.JPG")
csSirPic = fr.load_image_file("Mujibullah.JPG")
nagammalMamPic = fr.load_image_file("Nagammal R X C.JPG")
phySirPic = fr.load_image_file("Rajendran.JPG")
engMamPic = fr.load_image_file("Sandra Nigli - VII C.JPG")
tamilMamPic = fr.load_image_file("Saraswathi L Tamil Teacher.JPG")
sheejaMamPic = fr.load_image_file("Sheeja.JPG")
sivakamiMamPic = fr.load_image_file("Sivakami S - IX A.JPG")


myFaceEncodings = fr.face_encodings(myImage)[0]
mamFaceEncodings = fr.face_encodings(mamImage)[0]
gregorySirFaceEncodings = fr.face_encodings(gregorySirPic)[0]
beenaMamFaceEncodings = fr.face_encodings(beenaMamPic)[0]
deviMamFaceEncodings = fr.face_encodings(deviMamPic)[0]
yogaSirFaceEncodings = fr.face_encodings(yogaSir)[0]
csMamFaceEncodings = fr.face_encodings(csMamPic)[0]
csSirFaceEncodings = fr.face_encodings(csSirPic)[0]
nagammalMamFaceEncodings = fr.face_encodings(nagammalMamPic)[0]
phySirFaceEncodings = fr.face_encodings(phySirPic)[0]
engMamFaceEncodings = fr.face_encodings(engMamPic)[0]
tamilMamFaceEncodings = fr.face_encodings(tamilMamPic)[0]
sheejaMamFaceEncodings = fr.face_encodings(sheejaMamPic)[0]
sivakamiMamFaceEncodings = fr.face_encodings(sivakamiMamPic)[0]

known_face_encondings = [myFaceEncodings,mamFaceEncodings,gregorySirFaceEncodings,beenaMamFaceEncodings,deviMamFaceEncodings,yogaSirFaceEncodings,csMamFaceEncodings,csSirFaceEncodings,nagammalMamFaceEncodings,phySirFaceEncodings,engMamFaceEncodings,tamilMamFaceEncodings,sheejaMamFaceEncodings,sivakamiMamFaceEncodings]
known_face_names = ["Bargav","Correspondent mam","Gregory sir","Beena mam","Devi mam","Jayaraman sir","Manjula mam","Mujibullah sir","Nagammal mam","Rajendran sir","Sandra mam","Saraswathi mam","Sheeja mam","Sivakami mam"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[2].id)
            if name != "Correspondent mam":
                
                engine.say(f"hi {name}")
                engine.runAndWait()
            elif name == 'Correspondent mam':
                
                engine.say("namaste mam")
                engine.runAndWait()

            elif name=="Unknown":
                
                engine.say("face unknown")
                engine.runAndWait()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()