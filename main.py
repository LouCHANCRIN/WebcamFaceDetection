import cv2
import sys

def detectface(path):
    #Load classifier
    cascade = cv2.CascadeClassifier(path)

    #Use webcam
    video_capture = cv2.VideoCapture(0)
    print("Press q to quit")

    while True:
        #Read the image and turn it to grayscale for the classifier
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Finding the bounding box and adding them to the image
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30,30), flags=cv2.cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #Displaying the image
        cv2.imshow('Video', frame)

        #Keep going till user press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectface(sys.argv[1])
