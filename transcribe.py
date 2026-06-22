import whisper 
model = whisper.load_model("base")
result = model.transcribe("Recording.mp3")

print("Transcribed Text")
print(result["text"])

print("Detected Language")
print(result["language"])