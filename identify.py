import face_recognition
from PIL import Image, ImageDraw
import numpy as np

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
  "vikas"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('./img/group/team1.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
  print( "List of matches: ",matches)
  name = "Unknown Person"

  # If match
 # if True in matches:
  #  first_match_index = matches.index(True)
   # name = known_face_names[first_match_index]

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)

  print("index of best match: ", best_match_index)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]

  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('identify.jpg')