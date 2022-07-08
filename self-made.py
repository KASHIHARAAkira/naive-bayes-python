import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sudachipy import dictionary
from sudachipy import tokenizer
import numpy as np

##### _φ(･_･ 自作箇所 #####

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

# 事前確率（今回は全て1/3）
pC = [1/3, 1/3, 1/3]

# 形態素解析をするclass
class Morphology:
  def __init__(self, arr_title):
    self.data = arr_title
    self.cv = CountVectorizer(analyzer=self.split_words)
    self.cv.fit(arr_title)

    self.clf = MultinomialNB()
    self.xs = []

    for item in arr_title:
        self.xs.append(self.cv.transform([item]).toarray())
    X = np.concatenate(self.xs)
    self.clf.fit(X, [i for i in range(1, len(arr_title)+1)])

  def split_words(self, text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    return [m.surface() for m in tokenizer_obj.tokenize(text, mode)]

  def get_vocabulary(self):
    return self.cv.vocabulary_

  def get_words_frequency(self):
    return self.cv.transform(self.data).toarray()

  def get_matched_words(self, text):
    return self.cv.transform(text).toarray()

# 学習データを形態素解析するために初期化
morphology_learning = Morphology(learning)

# 学習データの単語一覧
words_list_learning = morphology_learning.get_vocabulary()

# 事前確率（単語の出現頻度）の定義
freq_learning = morphology_learning.get_words_frequency() # 各単語の出現頻度
total_each_class_learning = np.sum(freq_learning, axis=1) # 各クラスの単語数

# テストデータの情報を取得
morphology_testing = Morphology(testing)

# テストデータの単語一覧
words_list_testing = morphology_testing.get_vocabulary()

# テストデータの各単語出現頻度
freq_testing = morphology_testing.get_words_frequency() # 各単語の出現頻度
total_each_class_testing = np.sum(freq_testing, axis=1)  # 各クラスの単語数

# テストデータに出現する単語で学習データに含まれる単語を抽出
matched_words = morphology_learning.get_matched_words(testing)

print("📝自作単純ベイズの結果、()は類似度が高い順番。アルファベットはカテゴリ記号、『』は書名。")
for j in range(0,len(class_name)):
  result_arr = np.array([])
  for i in range(0,len(class_name)):

    num_arr_matched = np.intersect1d(np.nonzero(freq_learning[i]), np.nonzero(matched_words[j]))

    freq_learning_matched = freq_learning[i][num_arr_matched]  # テストデータで一致した学習データの出現頻度
    num_matched = matched_words[j][num_arr_matched]  # 学習データと一致したテストデータでの単語出現頻度

    len_non_zero_matched_words = np.sum(num_matched)  # テストデータで学習データと一致した単語の出現数（分子が0ではない要素の数）
    num_testing_words = total_each_class_testing[j] # テストデータの全体の単語数, 行列の長さになる
    len_zero_arr = num_testing_words - len_non_zero_matched_words # 要素がゼロの配列の長さ

    # 荷重スムージング
    # 非ゼロ要素の分子処理
    non_zero_smooth = freq_learning_matched + 1

    num = 0
    for item in num_matched:
      non_zero_smooth[num] = non_zero_smooth[num] ** item
      num = num + 1
    # ゼロの要素を足す
    pwc = np.append(non_zero_smooth, np.ones(len_zero_arr)) #分子が出来た

    # 分母を作る
    val_deno = np.sum(freq_learning[i]) # 分母となる基本の数（荷重スムージング前）
    len_learning = len(freq_learning[i]) # 学習データ全体に出現する単語から重複を抜いた数

    deno = np.array([])
    for item in num_matched:
      deno = np.append(deno, (val_deno + len_learning) ** item)
    deno = np.append(deno, np.full(len_zero_arr, (val_deno + len_learning)))
    members = pwc / deno
    result_arr = np.append(result_arr, np.sum(np.log(members)) + np.log(pC[i]))
  print("🌟【" + class_name[j] + "】「" + testing[j] + "」の類似度が高い順番")
  num = 1
  for item in np.argsort(result_arr)[::-1]:
    print("(" + str(num) + ") " + class_name[item])
    num = num + 1
  # print(result_arr) # 類似度（対数）
print("⌛実行時間⌛ " + str(time.time() - start_time) + "秒\n")
