from django.shortcuts import render
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from . import files
from moodchanger import settings
from .models import Audio


# Create your views here.
def home(request):
    return render(request,'base.html')

def play_audio(request,):
    title="songs/Maroon 5 - Memories (Official Video).mp3"
    #print(title)
    audio = Audio.objects.all().get(link=title)
    audio.link=settings.MEDIA_ROOT+"\Maroon 5 - Memories (Official Video).mp3"
    context = {
        'audio': audio
    }
    #print(audio)
    return render(request, 'myapp/test.html', context=context)


def facedetect(request):
    detection_model_path = settings.BASE_DIR+"\myapp\\files\haarcascade_frontalface_default.xml"
    emotion_model_path = settings.BASE_DIR+ "\myapp" + "\\files\_mini_XCEPTION.hdf5"


    cap = cv2.VideoCapture(0)

    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path)
    EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

    ret=1
    i=0
    while(ret!=0):
        
        ret, fm=cap.read()
        cv2.imwrite('live_test_img.jpeg', fm)
        
        img_path = 'live_test_img.jpeg'
        orig_frame = cv2.imread(img_path)
        frame = cv2.imread(img_path,0)
        faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) :
            faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
 
        #cv2.imshow('live_test_face', orig_frame)
        cv2.imwrite('final.jpeg', orig_frame)
        print(i,label)
        i=i+1
        k=cv2.waitKey(25)
        if k == 27 or i>10: 
            ret=0        # wait for ESC key to exit
            break

        #print("closed")
    cap.release()    
    cv2.destroyAllWindows()
        
    return render(request,'result.html',{'label':label})
    
    