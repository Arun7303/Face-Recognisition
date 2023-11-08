import cv2
import face_recognition
import os



def takeData():
    pass

# Function to save an image to the "images" folder
def save_image(image, folder, filename):
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, filename)
    cv2.imwrite(image_path, image)

# Function to load and encode all images in the "images" folder
def load_images_and_encodings(folder):
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    known_encodings = []

    for image_file in image_files:
        image = face_recognition.load_image_file(os.path.join(folder, image_file))
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append((encoding, image_file))

    return known_encodings

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Lower the resolution
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height

skip_frames = 2  # Skip processing every 2 frames
frame_counter = 0

known_encodings = load_images_and_encodings("images")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1

    if frame_counter % skip_frames != 0:
        continue

    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with known face encodings
        match = False
        for known_encoding, image_filename in known_encodings:
            result = face_recognition.compare_faces([known_encoding], face_encoding)
            if result[0]:
                color = (0, 255, 0)  # Green for a matched face
                name = image_filename
                match = True
                break

        if not match:
            color = (0, 0, 255)  # Red for an unmatched face
            name = "Unknown"

        # Draw a rectangle around the face with the appropriate color
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display the name and matching status
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the video frame
    cv2.imshow("Face Recognition", frame)

    # Press 'c' to capture and save an image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        save_image(frame, "images", f"captured_{frame_counter}.jpg")

    # Press 'q' to exit the loop
    if key == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()