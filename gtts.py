from gtts import gTTS
import playsound
import os

text = "The brain is responsible for cognition."
tts = gTTS(text)
filename = 'gtts.mp3'
tts.save(filename)
playsound.playsound(filename)
os.remove(filename)
