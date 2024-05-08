from underthesea import sent_tokenize
from underthesea import word_tokenize
import re
import roman
from vinorm import TTSnorm
# Mapping model
model_mapping={ 
    'nữ 3':'model_50h',
    'nữ 1': 'model_Lan_ttsmarker',
    'nữ 2' : 'model_numienbac',
    'nam 2' : 'model_40h',
    'nam 1' : 'model_nam_ttsmaker'
    
}

vietdict = {'b': 'bờ','c':'cờ', 'k': 'cờ', 'tr': 'chờ', 'd': 'dờ', 'đ': 'đờ', 'g': 'gờ', 'l': 'lờ', 'm': 'mờ','f':'phờ',
            'n': 'nờ', 'p': 'pờ', 'ph': 'phờ', 'r': 'rờ', 's': 'xờ', 't': 'tờ', 'th': 'thờ', 'v': 'vờ','x':'sờ'}
def short_dict():
	d = {}
	with open("customwords.txt",'r',encoding='utf-8') as f:
		for line in f:
			item = line.split(",")
			d[str(item[0]).lower()] = str(item[1].strip())
	return d

def vietnamese_dict():
        with open("vietnamese_dictionary.txt",'r',encoding='utf-8') as f:
            d = f.readlines()
        return [str(item.split('\n')[0]).lower() for item in d]

def convert_abbreviation_to_title(abbreviation:str):
    abbreviation_dict = {
        "PGS": "Phó giáo sư",
        "TS": "Tiến sĩ",
        "THS":"Thạc sĩ",
        "ThS":"Thạc sĩ",
        "BS": "Bác sĩ",
        "BSNT":"Bác sĩ nội trú",
        "GS":"Giáo sư",
        "CKI":"Chuyên khoa một",
        "CKII":"Chuyên khoa hai"
    }
    try:
        if abbreviation[-1]=='.':
            if not abbreviation[:-1].isalpha():
                return abbreviation
            else:
                return abbreviation[:-1]+' .'
        words = abbreviation.split(".")
        for i in range(len(words)):
            if not words[i].isalpha():
                return abbreviation
            if  words[i] in abbreviation_dict:
                words[i] = abbreviation_dict[words[i]]
        full_title = " ".join(words)
        return full_title
    except:
      return abbreviation
 
# custom_word = short_dict()
vietnamese_dictionary = vietnamese_dict()
exception_words={'AI':'Ây ai','Wi-Fi':'Goai phai','CEO':'Xê e ô'}

def map_customword(word:str,custom_word:dict):
    try:
       return custom_word[word.lower()]
    except:
      return word
    
def process_unvoice(english_word):
    sylable = []
    for i in english_word.split(" "):
        if i in vietdict:
            i = vietdict[i]
        sylable.append(i)
    return " ".join(sylable)
    
def pre_process_sentence(sentence):
    result = []
    sentence = sentence.strip()
    sentence = regex_ratio(sentence)
    sentence = regex_datemonth_missing(sentence)
    list_word = [word_tokenize(item) for item in word_tokenize(sentence)]
    list_word = flatten_comprehension(list_word)
    for word in list_word:
        char = re.findall(r'\w+|\S', word)
        if any(re.match(r'^(IX|IV|V?I{0,3})?$', item) for item in char):
            word = " ".join([str(chuyen_doi_so_la_ma(item)) for item in char])
        if word in exception_words:
            word = exception_words[word]
        word = convert_abbreviation_to_title(word)
        result.append(word)
    sentence = " ".join(result)
    sentence = re.sub(r'\s[/-]\s', '/',sentence)
    #sentence = TTSnorm(sentence)
    #sentence = sentence.lower()
    return sentence


ratio = ['\d{1,2}(\.\d{1,2})?\s{0,1}-\s{0,1}\d{1,2}(\.\d{1,2})?\s{0,1}(%|lần|ngày)']

def regex_ratio(txt):
    for patern in ratio:
        match = re.search(patern, txt)
        if match:
            replace_string = match.group().split('-')[0]+' đến '+ match.group().split('-')[1] 
            txt = re.sub(patern, replace_string, txt)
    return txt

def chuyen_doi_so_la_ma(s):
    if re.search(r'^(IX|IV|V?I{0,3})?$', s):
        return roman.fromRoman(s)
    else:
        return s
    
def convert_quarter_string(txt):
    patern = 'quý (\d\s{0,1})-(\d\s{0,1})/(\s{0,1}\d{4})'
    match = re.search(patern, txt)
    
    if match:
        start_quarter = match.group(1)
        end_quarter = match.group(2)
        year = match.group(3)
        txt = re.sub(patern, f'quý {start_quarter} đến {end_quarter} năm {year}', txt)
    return txt

def regex_datemonth_missing(txt):
    patern = '^([1-9]|[12][0-9]|3[01])\/([1-9]|1[0-2])(?!\S,)$'
    match = re.search(patern, txt)
    if match:
        replace_string = match.group().split('/')[0]+' tháng '+ match.group().split('/')[1] 
        txt = re.sub(patern, replace_string, txt)
    return txt 

def capitalize_sentence(text):
    list_sentences = sent_tokenize(text)
    return " ".join([sentence.capitalize() for sentence in list_sentences])

def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]