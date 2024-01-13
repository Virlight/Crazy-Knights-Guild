import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Using the microphone
with sr.Microphone() as source:
    print("Please speak, I am listening...")

    # Continuously listen
    while True:
        try:
            # Adjust the recognizer's ambient noise level and record the audio
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            print("listening ends, start analysis...")

            # Use Google speech recognition to convert the speech to text
            text = r.recognize_google(audio, language="en-US")
            print("You said:", text)

        except sr.UnknownValueError:
            # Unable to understand the speech
            print("Could not understand what you said.")
        except sr.RequestError:
            # API request error
            print("Could not obtain results from Google Speech Recognition service.")