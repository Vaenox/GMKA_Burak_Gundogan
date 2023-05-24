### kütüphaneleri tanımlayınız. ### 
import pandas as pd
from gensim.models import Word2Vec


### tanımlanan fonksiyonlar da pass'lı ifadeler eksiktir. Bu fonksiyon içeriklerini doldurunuz ###

# numerik karakterlerin kaldırılması
def remove_numeric(value):
    gecici=[item for item in value if not item.isdigit()]
    gecici="".join(gecici)
    return gecici


# emojilerin kaldırılması
def remove_emoji(value):
    gecici=re.compile("[\U00010000-\U0010ffff]",flags=re.UNICODE)
    gecici=gecici.sub(r"",value)
    return gecici

#noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r"[^\w\s]","",value)

#tek karakterli ifadelerin kaldırılması
def remove_single_chracter(value):
    return re.sub(r"(?:^| )\w(?:$| )","",value)

#linklerin kaldırılması 
def remove_link(value):
    return re.sub(r"((www.\.[^\s]+)|(https?://[^\s]+))","", value)

# hashtaglerin kaldırılması
def remove_hashtag(value):
    return re.sub(r"#[^\s]+","",value)

# kullanıcı adlarının kaldırılması
def remove_username(value):
    return re.sub("@[^\s]+","",value)


#kök indirgeme ve stop words işlemleri
def stem_word(value):
    stemmer=snowballstemmer.stemmer("turkish")
    value=value.lower()
    value=stemmer.stemWords(value.split())
    stopWords=["acaba","ama","aslında","az","bazı","belki","biri","birkaç","birşey","biz","bu","çok","çünkü","da","daha","de","defa","diye","eğer","en",
               "gibi","hem","hep","hepsi","nerde","nerede","nereye","niçin","niye","o","sanki","şey","siz","şu","tüm","ve","veya","ya","yani","bir","iki",
               "üç","dört","beş","altı","yedi","sekiz","dokuz","on"]
    
    value=[item for item in value if not item in stopWords]
    value=" ".join(value)
    return value

# ön işlem fonksiyonlarının sırayla çağırılması
def pre_processing(value):
    return [removeNumeric(removeEmoji(
        removeSingleCharacter(
        removeNoktalama(
        removeLink(
        removeHashtag(
        removeUsername(
        stemWord(word)
        ))))))) for word in value.split()]

# Boşlukların kaldırılması
def remove_space(value):
    return [item for item in value if item.strip()]

# word2vec model oluşturma ve kaydetme
def word2vec_create(value):
    model = Word2Vec(sentences = value.tolist(),vector_size=100,window=5,min_count=1)
    model.save("nlp_odev/data/word2vec.model")

# word2vec model yükleme ve vektör çıkarma
def word2vec_analysis(value):
    model = Word2Vec.load("nlp_odev/data/word2vec.model")
    geciciList=[]
    geciciLen=len(value)

    for k in value:
        gecici=model.wv.key_to_index[k]
        gecici=model.wv[gecici]
        geciciList.append(gecici)

    geciciList=sum(geciciList)
    geciciList=geciciList/geciciLen
    return geciciList.tolist()

# word2vec model güncellenir.
def word2vec_update(value):
    model = Word2Vec.load("nlp_odev/data/word2vec.model")
    model.build_vocab(value.tolist(),update=True)
    model.save("nlp_odev/data/word2vec.model")


if __name__ == '__main__':
   
    # veri temizlemesi için örnek veri kümemiz okunur.
    df_1 = pd.read_csv("nlp_odev/data/nlp.csv",index_col=0)


    ### tanımlanan df_1 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_1 içerisinde yeni bir sütun oluşturun. ###
    df_1["Text_2"]=df_1["Text"].apply(pre_processing)
    df_1["Text_2"]=df_1["Text_2"].apply(remove_space)

    ### df_1 içerisinde Text_2 sütununda boş liste kontrolü ###
    df_1[df_1["Text_2"].str[0].isnull()]

    df_1_index=df_1[df_1["Text_2"].str[0].isnull()].index
    df_1=df_1.drop(df_1_index)
    df_1=df_1.reset_index()
    del df_1["index"]

    
    ### word2vec model oluşturma ###
    model=Word2Vec(sentences=df["Text_2"].tolist(),vector_size=100,window=5,min_count=1)
    model.save("nlp_odev/data/word2vec.model")
    
    # df_1 dataframe mizi artık kullanmaycağımızdan ram de yer kaplamaması adına boş bir değer ataması yapıyoruz.
    df_1 = {}

    #############################################################################################################################################

    # sınıflandırma yapacağımız veri okunur.
    df_2 = pd.read_csv("nlp_odev/data/metin_siniflandirma.csv",index_col=0)

    ### tanımlanan df_2 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_2 içerisinde yeni bir sütun oluşturun. ###
    df_2["Text_2"]=df_2["Text"].apply(pre_processing)
    df_2["Text_2"]=df_2["Text_2"].apply(remove_space)
    
    ### df_2 içerisinde Text_2 sütununda boş liste kontrolü ###
    df_2[df_2["Text_2"].str[0].isnull()]

    df_2_index=df_2[df_2["Text_2"].str[0].isnull()].index
    df_2=df_2.drop(df_1_index)
    df_2=df_2.reset_index()
    del df_2["index"]

    ### sınıflandırma yapacağımız df_2 içerisinde bulunan Text_2 sütun verisini word2vec verisinde güncelleyin. ### 

    model=Word2Vec.load("nlp_odev/data/word2vec.model")

    model.build_vocab(df_2["Text 2"].to_list(),update=True) #yeni veri kümemizi modelin içine ekledik.

    model.save("nlp_odev/data/word2vec.model")


    ### Text_2 sütun üzerinden word2vec adında bu modeli kullanarak yeni bir sütun yaratın
    df["Word2Vec"]=df["Text 2"].apply(word2vec_create)

    ### word2vec sütunumuzu train test olarak bölün ###
    X_train,X_test,y_train,y_test=train_test_split(df_2["Word2Vec"].to_list(),df_2["Label"].to_list(),test_size=0.2,random_state=42)

    ### svm pipeline oluştur, modeği eğit ve test et ###
    svm=Pipeline([("SVM",LinearSVC())]).fit(X_train,y_train)

    y_pred=svm.predict(X_test)
    
    ### accuracy ve f1 score çıktısını print ile gösterin. ###
    print(accuracy_score(y_test,y_pred))

    print(f1_score(y_test,y_pred,average="weighted"))
               
               