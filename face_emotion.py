import cv2
from deepface import DeepFace
MULTIPLER = 0.2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# get webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

count = 0
run = True
while run:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesPos = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in facesPos:
        x1 = int(x-(w*MULTIPLER))
        y1 = int(y-(h*MULTIPLER))
        if (x1 < 0):
            x1 = 0
        if (y1 < 0):
            y1 = 0
        height, width, channels = frame.shape
        x2 = int(x+(w*(1+MULTIPLER)))
        y2 = int(y+(h*(1+MULTIPLER)))
        if (x2 > width):
            x2 = width
        if (y2 > height):
            y2 = height
        print (x1,y1,x2,y2)
        
        cropped_image = frame[y1:y2, x1:x2]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        try:
            result = DeepFace.analyze(cropped_image, actions = ['emotion'])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
            frame,
            result['dominant_emotion'],
            (x1, y1),
            font, 1,
            (0, 0, 255),
            2,
            cv2.LINE_4) 
                   
        except ValueError:
            print ("Can't find face.")
            
        except TypeError:
            print ("NoneType Error???") 

    cv2.imshow('Demo video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    count += 1
    if count >= 3000:
        run = False
    

cap.release()
cv2.destroyAllWindows()
