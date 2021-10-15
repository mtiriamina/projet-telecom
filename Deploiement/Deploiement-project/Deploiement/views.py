from django.http import StreamingHttpResponse, HttpResponseServerError, response
from django.views.decorators import gzip
import joblib
from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import pickle
import os
import cv2
import time
import imutils

import warnings

names = []
warnings.filterwarnings('ignore')



def face(request):
    global name
    curr_path = os.getcwd()

    # print("Loading face detection model")
    proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
    model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
    face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

    # print("Loading face recognition model")
    recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
    face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

    recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
    le = pickle.loads(open('le.pickle', "rb").read())
    print("Starting test video file")
    vs = cv2.VideoCapture(2)
    time.sleep(1)
    tries = 0
    access = False
    name=""
    while tries < 5:

        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False,
                                           False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.7:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                (fH, fW) = face.shape[:2]

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

    if name != "unknown":
        return render(request, "bi.html", {'name': name})
    else:
        return redirect("login")


def prediction(request):
    if request.method == 'POST':

        orange = request.POST['orange']
        ooredoo = request.POST['ooredoo']
        telecom = request.POST['telecom']
        secteur = request.POST['secteur']
        ville = request.POST['ville']
        offre = request.POST['offre']
        operateur = request.POST['operateur']
        periode = request.POST['periode']
        note = joblib.load("note.sav")
        appel = joblib.load("appel.sav")
        roaming = joblib.load("roaming.sav")
        internet = joblib.load("internet.sav")
        X = joblib.load("features.sav")
        preds = X.copy()
        for col in preds.columns:
            preds[col].values[:] = 0
        preds = preds.head(1)
        if orange != "null":
            preds[orange] = 1
        elif ooredoo != "null":
            preds[ooredoo] = 1
        elif telecom != "null":
            preds[telecom] = 1
        else:
            pass
        preds[[secteur, ville, offre, operateur, periode]] = 1
        predNote = note.predict(preds)
        predAppel = appel.predict(preds)
        predRoaming = roaming.predict(preds)
        predInternet = internet.predict(preds)

        if request.is_ajax():
            return JsonResponse({'predNote': predNote[0],
                                 'predAppel': predAppel[0],
                                 'predRoaming': predRoaming[0],
                                 'predInternet': predInternet[0]})

    return render(request, "forms.html")


def bi(request):
    if request.user.is_authenticated:
        return render(request, "bi.html")
    else:
        return redirect("login")


def speech(request):
    if request.user.is_authenticated:
        import speech_recognition as sr

        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        from transformers import pipeline
        tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
        model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

        nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        wav = []
        directory = r'voicewav'
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):

                r = sr.Recognizer()
                with sr.AudioFile(os.path.join(directory, filename)) as source:
                    audio = r.record(source)
                    try:
                        text = r.recognize_google(audio, language="fr-FR")
                        wav.append(text)
                    except:
                        print("Error")
            else:
                continue

        import pandas as pd
        feedback = []
        pred = []
        index = []
        for i in range(len(wav)):
            feedback.append(wav[i])
            index.append(i)
            pred.append(nlp(wav[i])[0]["label"])
        dfSpeech = pd.DataFrame(zip(feedback, pred), columns=['feedback', 'pred'])
        allData = []
        for i in range(dfSpeech.shape[0]):
            temp = dfSpeech.loc[i]
            allData.append(dict(temp))
        context = {'data': allData}

        return render(request, 'speech.html', context)
    else:
        return redirect("login")
