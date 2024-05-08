import json
import math
import time
from pathlib import Path
from enum import Enum
import os
import wave
import numpy as np
import onnxruntime
import hashlib
from typing import Iterable, List, Optional
import soundfile as sf
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run
from speexdsp_ns import NoiseSuppression


SPEED_VALUES = {"very_slow":1.5,
                "slow":1.2,
                "normal":1,
                "fast":0.6,
                "very_fast":0.4}
SAMPLE_RATE = 22050
NOISE_SCALE_W = 0.8
NOISE_SCALE = 0.667
PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence

def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm

def synthesize_ids_to_raw(
    phoneme_ids: List[int],
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
    model = None
) -> bytes:
    """Synthesize raw audio from phoneme ids."""

    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
    scales = np.array(
        [noise_scale, length_scale, noise_w],
        dtype=np.float32,
    )

    sid = None

    if speaker_id is not None:
        sid = np.array([speaker_id], dtype=np.int64)
    audio = model.run(
        None,
        {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
            "sid": sid,
        },
    )[0].squeeze((0, 1))
    # audio = denoise(audio, bias_spec, 10)
    audio = audio_float_to_int16(audio.squeeze())

    return audio.tobytes()

def synthesize(wav_file: wave.Wave_write,synthesize_stream_raw):
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setnchannels(1)  # mono

    for audio_bytes in synthesize_stream_raw:
        wav_file.writeframes(audio_bytes)

def synthesize_stream_raw(
    text: str,
    config,
    model,
    speed,
    sentence_silence: float = 0.0,
) -> Iterable[bytes]:
    """Synthesize raw audio per sentence from text."""
    sentence_phonemes = phonemize(config, text)
    phoneme_ids = []
    num_silence_samples = int(sentence_silence * SAMPLE_RATE)
    silence_bytes = bytes(num_silence_samples * 2)

    for phonemes in sentence_phonemes:
        phoneme_ids = phonemes_to_ids(config,phonemes)
        yield synthesize_ids_to_raw(
            phoneme_ids,
            speaker_id=None,
            length_scale=float(SPEED_VALUES[speed]),
            noise_scale=NOISE_SCALE,
            noise_w=NOISE_SCALE_W,
            model = model
        ) + silence_bytes

def text_to_speech(text:str,speed:str,model_name:str,text_hash:str):
    """Main entry point"""
    import time
    t0 = time.time()
    speed = speed.strip()
    text = text.strip()
    #text_hash = hashlib.sha1((text.lower()+speed+model_name).encode('utf-8')).hexdigest()
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(model_name, sess_options=sess_options)
    config = load_config(model_name)
    t1 = time.time()
    print(f"Inference load model tts: {t1-t0}s") 
    wav_path = os.path.join(cache_dir, f"{text_hash}.wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        synthesize(wav_file,synthesize_stream_raw(text,config,model,speed,0.5))
    ## OGG format file
    data, samplerate = sf.read(wav_path)
    os.remove(wav_path)
    wav_path = os.path.join(cache_dir, f"{text_hash}.ogg")
    sf.write(wav_path, data, samplerate)
    return wav_path



def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm

class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    TEXT = "text"

def phonemize(config, text: str) -> List[List[str]]:
    """Text to phonemes grouped by sentence."""
    if config["phoneme_type"] == PhonemeType.ESPEAK:
        if config["espeak"]["voice"] == "ar":
            # Arabic diacritization
            # https://github.com/mush42/libtashkeel/
            text = tashkeel_run(text)
        return phonemize_espeak(text, config["espeak"]["voice"])
    if config["phoneme_type"] == PhonemeType.TEXT:
        return phonemize_codepoints(text)
    raise ValueError(f'Unexpected phoneme type: {config["phoneme_type"]}')


def phonemes_to_ids(config, phonemes: List[str]) -> List[int]:
    """Phonemes to ids."""
    id_map = config["phoneme_id_map"]
    ids: List[int] = list(id_map[BOS])
    for phoneme in phonemes:
        if phoneme not in id_map:
            print("Missing phoneme from id map: %s", phoneme)
            continue
        ids.extend(id_map[phoneme])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids

def load_config(model):
    with open(f"{model}.json", "r") as file:
        config = json.load(file)
    return config

def denoise(input_path,output_path):
    frame_size = 256
    near = wave.open(input_path, 'rb')
    out = wave.open(output_path, 'wb')
    out.setnchannels(near.getnchannels())
    out.setsampwidth(near.getsampwidth())
    out.setframerate(near.getframerate())    
    noise_suppression = NoiseSuppression.create(frame_size, near.getframerate())

    in_data_len = frame_size
    in_data_bytes = frame_size * 2

    while True:
        in_data = near.readframes(in_data_len)
        if len(in_data) != in_data_bytes:
            break

        in_data = noise_suppression.process(in_data)

        out.writeframes(in_data)

    near.close()
    out.close()
    os.remove(input_path)
    return output_path