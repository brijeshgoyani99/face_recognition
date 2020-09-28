import face_recognition

image_of_brij = face_recognition.load_image_file('./img/known/brijesh5.jpeg')
brij_face_encoding = face_recognition.face_encodings(image_of_brij)[0]

unknown_image = face_recognition.load_image_file(
    './img/unknown/brijesh2.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [brij_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is Brijesh...')
else:
    print('This is NOT Brijesh....')