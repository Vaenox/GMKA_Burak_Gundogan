import pandas as pd
import re
import snowballstemmer




#sayısal değerlerin kaldırılması

def removeNumeric(value):
    gecici=[item for item in value if not item.isdigit()]
    gecici="".join(gecici)
    return gecici


#emojilerin kaldırılması

def removeEmoji(value):
    gecici=re.compile("[\U00010000-\U0010ffff]",flags=re.UNICODE)
    gecici=gecici.sub(r"",value)
    return gecici


# tek karakterli ifadelerin kaldırılması (harfler)

def removeSingleCharacter(value):
    return re.sub(r"(?:^| )\w(?:$| )","",value)



# noktalama işaretlerinin kaldırılması

def removeNoktalama(value):
    return re.sub(r"[^\w\s]","",value)


#linklerin kaldırılması

def removeLink(value):
    return re.sub(r"((www.\.[^\s]+)|(https?://[^\s]+))","", value)



# hashtag lerin kaldırılması

def removeHashtag(value):
    return re.sub(r"#[^\s]+","",value)





# kullanıcı adları kaldırılması

def removeUsername(value):
    return re.sub("@[^\s]+","",value)



# kök indirgeme ve stop word işlemleri
def stemWord(value):
    stemmer=snowballstemmer.stemmer("turkish")
    value=value.lower()
    value=stemmer.stemWords(value.split())
    stopWords=["acaba","ama","aslında","az","bazı","belki","biri","birkaç","birşey","biz","bu","çok","çünkü","da","daha","de","defa","diye","eğer","en",
               "gibi","hem","hep","hepsi","nerde","nerede","nereye","niçin","niye","o","sanki","şey","siz","şu","tüm","ve","veya","ya","yani","bir","iki",
               "üç","dört","beş","altı","yedi","sekiz","dokuz","on"]
    
    value=[item for item in value if not item in stopWords]
    value=" ".join(value)
    return value




#fonksiyonları çağırma
def preProcessing(value):
    return [removeNumeric(removeEmoji(
        removeSingleCharacter(
        removeNoktalama(
        removeLink(
        removeHashtag(
        removeUsername(
        stemWord(word)
        ))))))) for word in value.split()]


#Boşlukların Kaldırılması

def removeSpace(value):
    return [item for item in value if item.strip()]
