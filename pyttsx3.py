import pyttsx3
import ipdb
import os

engine = pyttsx3.init(driverName='nsss')
engine.setProperty('voice', 'com.apple.voice.compact.en-ZA.Tessa')
engine.setProperty('rate', 150)
full_path = os.path.join(os.getcwd(), "rnm.wav")
text = "The brain is responsible for cognition."
engine.save_to_file(text, full_path)
engine.runAndWait()
# ipdb.set_trace(context=5)

voices = engine.getProperty('voices')
for voice in voices:
    if any(element in ['en_US', 'en_GB', 'en_ZA', 'en_IN', 'en_AU', 'en_IE'] for element in voice.languages):
        print(f'{voice.gender} : {voice.languages} : {voice.id}')
        if ('com.apple.voice.compact' in voice.id):
            engine.setProperty('voice', voice.id)
            engine.say("The Brain is responsible for Cognition, it is the support of Thought and Mind.")
            engine.runAndWait()

engine.stop()
