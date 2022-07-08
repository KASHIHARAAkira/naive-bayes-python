import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sudachipy import dictionary
from sudachipy import tokenizer
import numpy as np

start_time = time.time()

# 学習データ（上から、人間失格、十二支考、「女らしさ」とは何か）
learning = [
    "自分には、人間の生活というものが、見当つかないのです。自分は東北の田舎に生れましたので、汽車をはじめて見たのは、よほど大きくなってからでした。",
    "ドイツのマンハールト夥しく材料を集めて研究した所に拠れば、穀物の命は穀物と別に存し、時として或る動物、時として男、もしくは女、また小児の形を現わすというのが穀精の信念だ。",
    "私は女子が「妊娠する」という一事を除けば、男女の性別に由って宿命的に課せられている分業というものを見出すことが出来ません。",
]

# テストデータ（上から、人間失格、十二支考、「女らしさ」とは何か）
testing = [
    "自分の父は、東京に用事の多いひとでしたので、上野の桜木町に別荘を持っていて、月の大半は東京のその別荘で暮していました。",
    "しかるところ、黄色の衣を着、黄牛に車を牽かせて乗り、従者ことごとく黄色な人が通り掛かり、小児を見るとすなわち穀賊何故ここに坐し居るかと問うた。",
    "「女らしさ」というものは、要するに私のいわゆる「人間性」に吸収し還元されてしまうものです。",
]

# カテゴリ（クラス）名
class_name = [
    "A 『人間失格』",
    "B 『十二支考』",
    "C 『「女らしさ」とは何か』",
]
start = time.time()

class Similarity:
    def __init__(self, arr_title):
        self.cv = CountVectorizer(analyzer=self.split_words)
        self.cv.fit(arr_title)

        self.clf = MultinomialNB()
        self.xs = []

        for item in arr_title:
            self.xs.append(self.cv.transform([item]).toarray())
        X = np.concatenate(self.xs)
        self.clf.fit(X, [i for i in range(1, len(learning)+1)])

    def split_words(self, text):
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.C
        return [m.surface() for m in tokenizer_obj.tokenize(text, mode)]

    def get_vocabulary(self):
        return self.cv.vocabulary_

    def get_word_frequency(self, num):
        return self.cv.transform([learning[num]]).toarray()

    def get_words_frequency(self):
        return self.cv.transform(learning).toarray()

    def words_frequency(self, text):
      return self.cv.transform(text).toarray()

    def predict_proba(self, text):
        x = self.cv.transform([text]).toarray()
        return self.clf.predict_proba(x)

    def predict_log_proba(self, text):
        x = self.cv.transform([text]).toarray()
        return self.clf.predict_log_proba(x)


sim = Similarity(learning)
print("📚ライブラリ（scikit-learn）での実行結果、()は類似度が高い順番。アルファベットはカテゴリ記号、『』は書名。")

# sklearnを使って単純ベイズ（ナイーブベイズ）を適用
count = 0
for item in testing:
    result = sim.predict_log_proba(item)
    y = np.argsort(result)[::-1]
    print("🔵【"+ class_name[count] + "】「" + item + "」の類似度が高い順番")
    count = count + 1
    for ele in range(len(result)):
        e = np.argsort(result)[::-1][ele]
        num = 1
        for i in range(len(e)-1, -1, -1):
            print("(" + str(num) + ")" +
                  class_name[e[i]])
            num = num + 1

elapsed_time = time.time() - start
print("⏰実行時間⏰ " + str(time.time() - start) + "秒")
