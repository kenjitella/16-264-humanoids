import ollama
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import simpleaudio as sa
import wave
import io
from pydub import AudioSegment

elevenlabs_client = ElevenLabs(api_key='sk_0aace36375e9d05ff2e6cebca109bdc114024466157df056')
ollama_client = ollama.Client()





def main():
    n = 0
    model = "lia"
    prompt = "Introduce yourself say nothing about blocks"
    response = ollama_client.generate(model=model, prompt=prompt)
    print(response['response'])
    intro = text_to_speech_file(response['response'], n)
    play_audio(intro)
    while True:
        prompt = input(": ")
        response = ollama_client.generate(model=model, prompt=prompt)
        audiopath = text_to_speech_file(response['response'], n)
        play_audio(audiopath)
        print(response['response'])
        n += 1

def text_to_speech_file(text: str, n: int) -> str:
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="aEO01A4wXwd1O8GPgGlF",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2"
        )

    save_file_path = f"conversation{n}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"Audio saved: {save_file_path}")
    return save_file_path

def play_audio(file_path: str):
    audio = AudioSegment.from_file(file_path, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    wave_read = wave.open(wav_io, 'rb')
    wave_obj = sa.WaveObject.from_wave_read(wave_read)
    play_obj = wave_obj.play()
    play_obj.wait_done()

if __name__ == '__main__':
    main()
