from pydub import AudioSegment
import speech_recognition as sr
import os

# Load the audio file
audio_path = "Titanic.mp3"
audio_name, _ = os.path.splitext(audio_path)
audio = AudioSegment.from_file(audio_path)

# Convert to a format compatible with the speech recognition library
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export("data/converted_audio.wav", format="wav")

# Prepare the recognizer
r = sr.Recognizer()
converted_audio_path = "data/converted_audio.wav"

# Recognizing the speech in the audio file
with sr.AudioFile(converted_audio_path) as source:
    audio_listened = r.record(source)

# Trying to recognize the speech in English
try:
    text = r.recognize_google(audio_listened, language="en-US")
except sr.UnknownValueError:
    text = "Speech recognition could not understand the audio"
except sr.RequestError as e:
    text = f"Could not request results from Google Speech Recognition service; {e}"

print(text)
file_path = f"data/{audio_name}.txt"  
with open(file_path, "w") as file:
    file.write(text)
