import time
import os
import wave
import uvicorn
import hashlib
from fastapi import Body, Depends, FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi import  HTTPException
from rule import *
from tts import denoise
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Run at startup!")
    classifier = load_model()
    yield
    print("Run on shutdown!")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
   return {"message": "TTS API"}

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

@app.post("/process_text/")
async def process_text(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only text files (.txt) are allowed")

    contents = await file.read()

    processed_text = contents.decode("utf-8") 
    processed_text = processed_text.replace("\r\n\r\n", ". ")
    processed_text = processed_text.replace("\n", ". ")
    try:
        prompt=processed_text
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
        return {'audio_path': audio_path}
    except:
        raise HTTPException(status_code=404, detail="File not found")
    
@app.get("/audio/")
def predict(audio_path: str):
    try:
        return FileResponse(audio_path, media_type='audio/ogg') 
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    
def cleanup_cache():
    """
    Remove files in *basedir* not accessed within *limit* minutes

    :param basedir: directory to clean
    :param limit: minutes
    """
    basedir = os.path.join(os.getcwd(), "cache")
    os.makedirs(basedir, exist_ok=True)
    limit = 6 * 60 * 60
    atime_limit = time.time() - limit
    count = 0
    for filename in os.listdir(basedir):
        path = os.path.join(basedir, filename)
        if os.path.getatime(path) < atime_limit:
            os.remove(path)
            count += 1
    print("Removed {} files.".format(count))

# Schedule cleanup cache with file created >= 6 hours
scheduler = BackgroundScheduler()
# After 30 minutes check file in cache folder if file created >= 6 hours -> remove
scheduler.add_job(cleanup_cache, "interval", minutes=30)
scheduler.start()

# # Stop khi FastAPI stop
# @app.on_event("shutdown")
# def shutdown_event():
#     scheduler.shutdown()

# @app.get("/file")
# def get_audio(audio_path: str):
#     try:
#         return FileResponse(audio_path, media_type='audio/wav')
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="File not found")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5050)