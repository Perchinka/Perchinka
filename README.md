There is something

#выделение ключевых слов/биграмм
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
now_them={}
n = []
for i in train["Nominations"]:
    now_them[i] = 0
x = vectorizer.fit_transform(train["text"])
for i in range(len(x.agrsort.toarray()[:, :-30])):
    for i1 in i:
        for i in range(len(now_them)):
            now_them[now_them.dict_keys()[i]] = i1[i]
        for i in df_full_train["Nominmations"]:
            n.append(now_them[i])
        df_full_train[x.get_feature_names_out()[i1]] = n
#Тематическое моделирование
tfidf = TfidfVectorizer()
x1 = tfidf.fit_transform(df_full_train["text"])
clus = KMeans(n_clusters=50)
k = clus.fit(x1.toarray())
df_full_train.insert("themes", value=k.labels_, loc=len(df_full_train.columns))
