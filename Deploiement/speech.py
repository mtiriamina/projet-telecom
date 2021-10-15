import speech_recognition as sr
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')
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
for i in range(len(wav)):
    feedback.append(wav[i])
    pred.append(nlp(wav[i])[0]["label"])
dfSpeech = pd.DataFrame(zip(feedback, pred), columns=['feedback', 'pred'])
