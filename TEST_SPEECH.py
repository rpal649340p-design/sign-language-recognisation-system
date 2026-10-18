
import pyttsx3

for message in ["Yes", "No", "Stop", "Okay", "Peace", "Thank you", "Question"]:
    print("Speaking:", message)

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(message)
    engine.runAndWait()
    engine.stop()

print("Test finished.")
