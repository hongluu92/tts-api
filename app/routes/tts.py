from app.config.database import get_session
import time
import os
import wave
import hashlib
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi import  HTTPException
from rule import *
from tts import denoise
from fastapi import APIRouter, BackgroundTasks, Depends, status, Header
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import EncoderDecoderModel
from importlib.machinery import SourceFileLoader
from model_map.envibert_tokenizer import RobertaTokenizer
import torch
from contextlib import asynccontextmanager
from lingua import Language, LanguageDetectorBuilder
from functools import partial
import soundfile as sf
from voice import PiperVoice
from vinorm import TTSnorm
from app.config.security import get_current_user, oauth2_scheme
from app.config.database import get_session
from sqlalchemy.orm import Session
from app.config.security import get_current_user, oauth2_scheme

tts_router = APIRouter(
    prefix="/tts",
    tags=["TTS"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme), Depends(get_current_user)]
)
languages = [Language.ENGLISH, Language.VIETNAMESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

synthesize = None
cache_dir='model_map'
spell_tokenizer = RobertaTokenizer(pretrained_file=cache_dir)
spell_model = None

def oov_spelling(word, num_candidate=1):
    result = []
    inputs = spell_tokenizer([word.lower()])
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask)
    }
    outputs = spell_model.generate(**inputs, num_return_sequences=num_candidate)
    for output in outputs.cpu().detach().numpy().tolist():
        result.append(spell_tokenizer.sp_model.DecodePieces(spell_tokenizer.decode(output, skip_special_tokens=True).split()))
    return result   

def remove_silent_letter(word):
    letters = word.split()[::-1]
    try:
        word = " ".join(letters[next(index for index, value in enumerate(letters) if len(value) > 1 or value not in vietdict):][::-1])
    except:
        word
    return word

def preprocess_english_words(text):
    list_word = [word_tokenize(item) for item in word_tokenize(text)]
    list_word = flatten_comprehension(list_word)
    custom_word = short_dict()
    #print(list_word)
    for i in range(len(list_word)):
        if re.match('(?:[a-zA-ZÀ-ỹ]+(?:\s+[a-zA-ZÀ-ỹ]+)*)',list_word[i]):
            list_word[i] = map_customword(list_word[i],custom_word)
            if len(list_word[i]) == 1:
                continue
            if list_word[i] in custom_word.values():
                continue
            if list_word[i] in vietnamese_dictionary:
                continue
            lang = detector.detect_language_of(list_word[i])
            if lang != Language.VIETNAMESE:
                custom_word = short_dict()
                "Học những từ tiếng anh để cho các lần xử lý sau không phải học lại nữa"
                if list_word[i].lower() not in custom_word:
                    with open("customwords.txt",'a',encoding='utf-8') as f1: 
                        f1.write('\n'+list_word[i].lower()+',')
                        list_word[i] = oov_spelling(list_word[i])[0]
                        list_word[i] = remove_silent_letter(list_word[i])
                        list_word[i] = process_unvoice(list_word[i]) 
                        list_word[i] = list_word[i].capitalize()
                        f1.write(list_word[i])
                else:
                    list_word[i] = map_customword(list_word[i],custom_word)
    sentence = " ".join(list_word)
    sentence = TTSnorm(sentence)
    sentence = capitalize_sentence(sentence)
    return sentence[:-1]

def load_model():
    global spell_model
    global synthesize
    if spell_model:
        if synthesize:
            return spell_model,synthesize
    spell_model = EncoderDecoderModel.from_pretrained(pretrained_model_name_or_path='pytorch_model.bin',config='config.json')
    model = 'model_vbeetuongvi/model.onnx'
    speaker_id=None
    voice = PiperVoice(model)
    synthesize = partial(
        voice.synthesize,
        speaker_id=speaker_id)
    print("Model loaded.")
    return  spell_model,synthesize

def text_to_speech(text:str,text_hash:str):
    """Main entry point"""
    text = text.strip()
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True) 
    wav_path = os.path.join(cache_dir, f"{text_hash}.wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        synthesize(text,wav_file)
    ## OGG format file
    wav_path = denoise(wav_path,os.path.join(cache_dir, "denoise.wav"))
    data, samplerate = sf.read(wav_path)
    os.remove(wav_path)
    wav_path = os.path.join(cache_dir, f"{text_hash}.ogg")
    sf.write(wav_path, data, samplerate)
    return wav_path

@tts_router.get("/tts/")
def predict(prompt : str,user = Depends(get_current_user),session: Session = Depends(get_session)): 
    try:
        t0 = time.time()
        prompt = prompt.strip()
        text_hash = hashlib.sha1((prompt).encode('utf-8')).hexdigest()
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        list_file_in_cache = os.listdir(cache_dir)
        list_name_file_in_cache = [i.split('.')[0] for i in list_file_in_cache]
        if text_hash in list_name_file_in_cache:
            audio_path = os.path.join(cache_dir, f"{text_hash}.ogg")
        else:
            prompt = pre_process_sentence(prompt)
            t1 = time.time()
            print(f"Inference1: {t1-t0}s") 
            prompt = preprocess_english_words(prompt)
            t1 = time.time()
            print(f"Inference2: {t1-t0}s")
            print(f'Text processed:{prompt}') 
            audio_path = text_to_speech(prompt,text_hash)
        t1 = time.time()
        print(f"Inference3: {t1-t0}s") 
        return FileResponse(audio_path, media_type='audio/ogg') 
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")