import face_recognition
import cv2
#from PIL import Image, ImageDrawp
import numpy as np

video_capture = cv2.VideoCapture(0)

image_of_bill = face_recognition.load_image_file('./img/known/bill-gates1.jpeg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

image_of_steve = face_recognition.load_image_file('./img/known/brijesh5.jpeg')
brijesh_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

image_of_elon = face_recognition.load_image_file('./img/known/dhoni1.jpeg')
dhoni_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/kevin mitnick1.jpeg')
kevin_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/steve jobs1.jpeg')
steve_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/warren buffett1.jpeg')
warren_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/manav1.jpeg')
manav_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/parth1.jpeg')
parth_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/pratik1.jpeg')
pratik_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/dhruv1.jpeg')
dhruv_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/rutvik1.jpeg')
rutvik_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/tarang1.jpeg')
tarang_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/vikas1.jpeg')
vikas_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_elon = face_recognition.load_image_file('./img/known/himanshu1.jpeg')
himanshu_face_encoding = face_recognition.face_encodings(image_of_elon)[0]


#  Create arrays of encodings and names
known_face_encodings = [
  bill_face_encoding,
  brijesh_face_encoding,
  dhoni_face_encoding,
  kevin_face_encoding,
  steve_face_encoding,
  warren_face_encoding,
  manav_face_encoding,
  parth_face_encoding,
  pratik_face_encoding,
  dhruv_face_encoding,
  rutvik_face_encoding,
  tarang_face_encoding,
  vikas_face_encoding,
  himanshu_face_encoding
]

known_face_names = [
  "Bill Gates",
  "Brijesh",
  "MS Dhoni",
  "Kevin Mitnick",
  "Steve jobs",
  "Warren Buffett",
  "Manav",
  "parth",
  "pratik",
  "dhruv",
  "rutvik",
  "tarang",
  "vikas",
  "Himanshu"
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 5), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()