import time
import argparse
import sys

try:
    import whisper
except Exception as e:
    print("Missing dependency 'whisper'. Install with: pip install openai-whisper")
    sys.exit(1)


def transcribe(model_name: str, audio_path: str, device: str = None):
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    print(f"Transcribing with {model_name}...")
    t0 = time.time()
    result = model.transcribe(audio_path)
    t1 = time.time()
    elapsed = t1 - t0
    text = result.get("text", "")
    return {
        "model": model_name,
        "text": text.strip(),
        "elapsed": elapsed,
        "segments": result.get("segments", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Whisper base and small models on the same audio file")
    parser.add_argument("audio", help="Path to audio file to transcribe")
    parser.add_argument("--device", default=None, help="Device to run on (e.g. cpu or cuda)")
    args = parser.parse_args()

    audio = args.audio
    device = args.device

    models = ["base", "small"]
    results = []
    for m in models:
        try:
            res = transcribe(m, audio, device=device)
            results.append(res)
        except Exception as e:
            print(f"Error with model {m}: {e}")

    print("\n--- Results ---")
    for r in results:
        print(f"Model: {r['model']}")
        print(f"Time (s): {r['elapsed']:.2f}")
        print(f"Transcription: {r['text']}\n")

    if len(results) == 2:
        a = results[0]['text']
        b = results[1]['text']
        if a == b:
            print("Transcriptions are identical.")
        else:
            print("Transcriptions differ.\n")
            print("--- Diff ---")
            import difflib
            for line in difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=models[0], tofile=models[1], lineterm=""):
                print(line)


if __name__ == '__main__':
    main()
