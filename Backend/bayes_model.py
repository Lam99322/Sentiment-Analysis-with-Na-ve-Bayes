# import math
# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict, Counter

# # Tải dữ liệu NLTK cần thiết (chỉ lần đầu)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # ==========================
# #   SENTENCE PARSER NÂNG CAO
# # ==========================
# class SentenceParser:
#     def __init__(self, sentence):
#         self.sentence = sentence or ""
#         self.stop_words = set([
#             "", "i","me","my","myself","you","your","he","him","his",
#             "she","her","it","its","we","us","they","them","this","that",
#             "those","with","am","is","are","was","were","to","a","an","the",
#             "of","for","in","on","at","and","or","as","by","do","doing","from",
#             "have","has","had","be","been","being","there","here","when","where",
#             "who","which","because","due","after","before","let","make","get",
#             "out","still","that's","it's","your"
#         ])
#         self.lemmatizer = WordNetLemmatizer()

#     def parse(self, ngrams=1):
#         words = re.findall(r'\b\w+\b', self.sentence.lower())
#         words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
#         if ngrams > 1:
#             words = self._make_ngrams(words, ngrams)
#         return words

#     def _make_ngrams(self, words, n):
#         return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# # ==========================
# #   BAYES CLASSIFIER TF-IDF
# # ==========================
# class Bayes_Classifier:
#     def __init__(self):
#         self.pos_dic = defaultdict(float)
#         self.neg_dic = defaultdict(float)
#         self.vocab = set()
#         self.df = defaultdict(int)  # document frequency
#         self.doc_count = 0
#         self.pos_docs = 0
#         self.neg_docs = 0
#         self.pos_word_sum = 0
#         self.neg_word_sum = 0
#         self.prob_pos = 0.5
#         self.prob_neg = 0.5

#     # ------------------------
#     #   TRAINING
#     # ------------------------
#     def train(self, lines):
#         docs_words = []
#         total_valid = 0

#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             label, _, text = parts
#             words = SentenceParser(text).parse(ngrams=2)  # dùng bigram
#             if not words:
#                 continue

#             docs_words.append((label, words))
#             total_valid += 1

#             # Cập nhật doc frequency
#             unique_words = set(words)
#             for w in unique_words:
#                 self.df[w] += 1

#             if label == "5":
#                 self.pos_docs += 1
#             else:
#                 self.neg_docs += 1

#         self.doc_count = total_valid
#         self.prob_pos = self.pos_docs / total_valid if total_valid > 0 else 0.5
#         self.prob_neg = self.neg_docs / total_valid if total_valid > 0 else 0.5

#         # Tính TF-IDF cho từng từ
#         N = self.doc_count
#         for label, words in docs_words:
#             tf = Counter(words)
#             for w, count in tf.items():
#                 idf = math.log((N + 1) / (self.df[w] + 1)) + 1
#                 tf_idf = count * idf
#                 self.vocab.add(w)
#                 if label == "5":
#                     self.pos_dic[w] += tf_idf
#                     self.pos_word_sum += tf_idf
#                 else:
#                     self.neg_dic[w] += tf_idf
#                     self.neg_word_sum += tf_idf

#         self.total_words = len(self.vocab) or 1

#     # ------------------------
#     #   SCORE FUNCTION
#     # ------------------------
#     def _score(self, words, label):
#         s = math.log(self.prob_pos if label=="5" else self.prob_neg)
#         dic = self.pos_dic if label=="5" else self.neg_dic
#         total_tf_idf = self.pos_word_sum if label=="5" else self.neg_word_sum
#         V = self.total_words

#         for w in words:
#             # Laplace smoothing
#             s += math.log((dic.get(w, 0) + 1) / (total_tf_idf + V))
#         return s

#     # ------------------------
#     #   CLASSIFY SINGLE TEXT
#     # ------------------------
#     def classify_text(self, text):
#         words = SentenceParser(text).parse(ngrams=1)
#         p_score = self._score(words, "5")
#         n_score = self._score(words, "1")
#         label = "5" if p_score >= n_score else "1"

#         try:
#             prob_pos = 1 / (1 + math.exp(n_score - p_score))
#             prob_neg = 1 - prob_pos
#         except:
#             prob_pos, prob_neg = 0.5, 0.5

#         return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg}

#     # ------------------------
#     #   CLASSIFY MULTIPLE LINES
#     # ------------------------
#     def classify_lines(self, lines):
#         results = []
#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             _, _, text = parts
#             res = self.classify_text(text)
#             results.append(res["label"])
#         return results
# import math
# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict, Counter

# # Tải dữ liệu NLTK cần thiết
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # ==========================
# #   SENTENCE PARSER
# # ==========================
# class SentenceParser:
#     def __init__(self, sentence):
#         self.sentence = sentence or ""
#         self.stop_words = set([
#             "", "i","me","my","myself","you","your","he","him","his",
#             "she","her","it","its","we","us","they","them","this","that",
#             "those","with","am","is","are","was","were","to","a","an","the",
#             "of","for","in","on","at","and","or","as","by","do","doing","from",
#             "have","has","had","be","been","being","there","here","when","where",
#             "who","which","because","due","after","before","let","make","get",
#             "out","still","that's","it's","your"
#         ])
#         self.lemmatizer = WordNetLemmatizer()

#     def parse(self, ngrams=1):
#         words = re.findall(r'\b\w+\b', self.sentence.lower())
#         words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
#         if ngrams > 1:
#             words = self._make_ngrams(words, ngrams)
#         return words

#     def _make_ngrams(self, words, n):
#         return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# # ==========================
# #   BAYES CLASSIFIER TF-IDF
# # ==========================
# class Bayes_Classifier:
#     def __init__(self):
#         self.pos_dic = defaultdict(float)
#         self.neg_dic = defaultdict(float)
#         self.vocab = set()
#         self.df = defaultdict(int)  # document frequency
#         self.doc_count = 0
#         self.pos_docs = 0
#         self.neg_docs = 0
#         self.pos_word_sum = 0
#         self.neg_word_sum = 0
#         self.prob_pos = 0.5
#         self.prob_neg = 0.5

#     # ------------------------
#     #   TRAINING
#     # ------------------------
#     def train(self, lines):
#         docs_words = []
#         total_valid = 0

#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             label, _, text = parts
#             words = SentenceParser(text).parse(ngrams=2)  # dùng bigram
#             if not words:
#                 continue

#             docs_words.append((label, words))
#             total_valid += 1

#             # Cập nhật doc frequency
#             unique_words = set(words)
#             for w in unique_words:
#                 self.df[w] += 1

#             if label == "5":
#                 self.pos_docs += 1
#             else:
#                 self.neg_docs += 1

#         self.doc_count = total_valid
#         self.prob_pos = self.pos_docs / total_valid if total_valid > 0 else 0.5
#         self.prob_neg = self.neg_docs / total_valid if total_valid > 0 else 0.5

#         # Tính TF-IDF cho từng từ
#         N = self.doc_count
#         for label, words in docs_words:
#             tf = Counter(words)
#             for w, count in tf.items():
#                 idf = math.log((N + 1) / (self.df[w] + 1)) + 1
#                 tf_idf = count * idf
#                 self.vocab.add(w)
#                 if label == "5":
#                     self.pos_dic[w] += tf_idf
#                     self.pos_word_sum += tf_idf
#                 else:
#                     self.neg_dic[w] += tf_idf
#                     self.neg_word_sum += tf_idf

#         self.total_words = len(self.vocab) or 1

#     # ------------------------
#     #   SCORE FUNCTION
#     # ------------------------
#     def _score(self, words, label):
#         s = math.log(self.prob_pos if label=="5" else self.prob_neg)
#         dic = self.pos_dic if label=="5" else self.neg_dic
#         total_tf_idf = self.pos_word_sum if label=="5" else self.neg_word_sum
#         V = self.total_words

#         for w in words:
#             # Laplace smoothing
#             val = dic.get(w, 0) + 1
#             # Tăng trọng số từ negative nếu label là negative
#             if label=="1" and w in dic:
#                 val *= 2.5
#             s += math.log(val / (total_tf_idf + V))
#         return s

#     # ------------------------
#     #   CLASSIFY SINGLE TEXT
#     # ------------------------
#     def classify_text(self, text):
#         words = SentenceParser(text).parse(ngrams=1)  # unigram
#         p_score = self._score(words, "5")
#         n_score = self._score(words, "1")
#         label = "5" if p_score >= n_score else "1"

#         try:
#             prob_pos = 1 / (1 + math.exp(n_score - p_score))
#             prob_neg = 1 - prob_pos
#         except:
#             prob_pos, prob_neg = 0.5, 0.5

#         return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

#     # ------------------------
#     #   CLASSIFY MULTIPLE LINES
#     # ------------------------
#     def classify_lines(self, lines):
#         results = []
#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             _, _, text = parts
#             res = self.classify_text(text)
#             results.append(res)
#         return results

# # ==========================
# #   TEST NHANH
# # ==========================
# if __name__ == "__main__":
#     # Ví dụ dataset nhỏ
#     training_data = [
#         "5|0|I love this product",
#         "5|0|This is amazing",
#         "1|0|I hate this",
#         "1|0|Very sad experience",
#         "1|0|Terrible outcome",
#         "5|0|Fantastic work"
#     ]

#     classifier = Bayes_Classifier()
#     classifier.train(training_data)

#     # Test câu ngắn
#     test_texts = ["sad", "I am very sad today", "fantastic", "I hate it"]
#     for t in test_texts:
#         res = classifier.classify_text(t)
#         print(t, res)
# import math
# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict, Counter

# # Tải dữ liệu NLTK cần thiết (chỉ lần đầu)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # ==========================
# # SENTENCE PARSER
# # ==========================
# class SentenceParser:
#     def __init__(self, sentence):
#         self.sentence = sentence or ""
#         self.stop_words = set([
#             "", "i","me","my","myself","you","your","he","him","his",
#             "she","her","it","its","we","us","they","them","this","that",
#             "those","with","am","is","are","was","were","to","a","an","the",
#             "of","for","in","on","at","and","or","as","by","do","doing","from",
#             "have","has","had","be","been","being","there","here","when","where",
#             "who","which","because","due","after","before","let","make","get",
#             "out","still","that's","it's","your"
#         ])
#         self.lemmatizer = WordNetLemmatizer()

#     def parse(self, ngrams=1):
#         words = re.findall(r'\b\w+\b', self.sentence.lower())
#         words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
#         if ngrams > 1:
#             words = self._make_ngrams(words, ngrams)
#         return words

#     def _make_ngrams(self, words, n):
#         return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# # ==========================
# # BAYES CLASSIFIER TF-IDF + DICTIONARY
# # ==========================
# class Bayes_Classifier:
#     def __init__(self):
#         self.pos_dic = defaultdict(float)
#         self.neg_dic = defaultdict(float)
#         self.vocab = set()
#         self.df = defaultdict(int)
#         self.doc_count = 0
#         self.pos_docs = 0
#         self.neg_docs = 0
#         self.pos_word_sum = 0
#         self.neg_word_sum = 0
#         self.prob_pos = 0.5
#         self.prob_neg = 0.5

#         # Dictionary cảm xúc để nhận từ đơn lẻ
#         self.emotion_dict = {
#             "love": "5",
#             "happy": "5",
#             "amazing": "5",
#             "fantastic": "5",
#             "great": "5",
#             "good": "5",
#             "excellent": "5",
#             "hate": "1",
#             "sad": "1",
#             "terrible": "1",
#             "bad": "1",
#             "awful": "1",
#             "horrible": "1"
#         }

#     # ------------------------
#     # TRAINING
#     # ------------------------
#     def train(self, lines):
#         docs_words = []
#         total_valid = 0

#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             label, _, text = parts
#             words = SentenceParser(text).parse(ngrams=1)  # unigram
#             if not words:
#                 continue

#             docs_words.append((label, words))
#             total_valid += 1

#             unique_words = set(words)
#             for w in unique_words:
#                 self.df[w] += 1

#             if label == "5":
#                 self.pos_docs += 1
#             else:
#                 self.neg_docs += 1

#         self.doc_count = total_valid
#         self.prob_pos = self.pos_docs / total_valid if total_valid > 0 else 0.5
#         self.prob_neg = self.neg_docs / total_valid if total_valid > 0 else 0.5

#         N = self.doc_count
#         for label, words in docs_words:
#             tf = Counter(words)
#             for w, count in tf.items():
#                 idf = math.log((N + 1) / (self.df[w] + 1)) + 1
#                 tf_idf = count * idf
#                 self.vocab.add(w)
#                 if label == "5":
#                     self.pos_dic[w] += tf_idf
#                     self.pos_word_sum += tf_idf
#                 else:
#                     self.neg_dic[w] += tf_idf
#                     self.neg_word_sum += tf_idf

#         self.total_words = len(self.vocab) or 1

#     # ------------------------
#     # SCORE
#     # ------------------------
#     def _score(self, words, label):
#         s = math.log(self.prob_pos if label=="5" else self.prob_neg)
#         dic = self.pos_dic if label=="5" else self.neg_dic
#         total_tf_idf = self.pos_word_sum if label=="5" else self.neg_word_sum
#         V = self.total_words

#         for w in words:
#             val = dic.get(w, 0) + 1
#             s += math.log(val / (total_tf_idf + V))
#         return s

#     # ------------------------
#     # CLASSIFY SINGLE TEXT
#     # ------------------------
#     def classify_text(self, text):
#         words = SentenceParser(text).parse(ngrams=1)

#         # Kiểm tra dictionary trước
#         for w in words:
#             if w in self.emotion_dict:
#                 label = self.emotion_dict[w]
#                 prob_pos = 1.0 if label=="5" else 0.0
#                 prob_neg = 1.0 - prob_pos
#                 return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

#         # Nếu không có trong dictionary, dùng Bayes TF-IDF
#         p_score = self._score(words, "5")
#         n_score = self._score(words, "1")
#         label = "5" if p_score >= n_score else "1"

#         try:
#             prob_pos = 1 / (1 + math.exp(n_score - p_score))
#             prob_neg = 1 - prob_pos
#         except:
#             prob_pos, prob_neg = 0.5, 0.5

#         return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

# # ==========================
# # TEST NHANH
# # ==========================
# if __name__ == "__main__":
#     # Dataset huấn luyện nhỏ
#     training_data = [
#         "5|0|I love this product",
#         "5|0|This is amazing",
#         "5|0|Fantastic work",
#         "5|0|happy",
#         "1|0|I hate this",
#         "1|0|Very sad experience",
#         "1|0|Terrible outcome",
#         "1|0|sad"
#     ]

#     classifier = Bayes_Classifier()
#     classifier.train(training_data)

#     # Test từ/câu ngắn
#     test_texts = ["sad", "I am very sad today", "fantastic", "I hate it", "love", "happy"]
#     for t in test_texts:
#         res = classifier.classify_text(t)
#         print(f"Input: {t}\nOutput: {res}\n")
# import math
# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict, Counter

# # Tải dữ liệu NLTK cần thiết
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # ==========================
# # SENTENCE PARSER
# # ==========================
# class SentenceParser:
#     def __init__(self, sentence):
#         self.sentence = sentence or ""
#         self.stop_words = set([
#             "", "i","me","my","myself","you","your","he","him","his",
#             "she","her","it","its","we","us","they","them","this","that",
#             "those","with","am","is","are","was","were","to","a","an","the",
#             "of","for","in","on","at","and","or","as","by","do","doing","from",
#             "have","has","had","be","been","being","there","here","when","where",
#             "who","which","because","due","after","before","let","make","get",
#             "out","still","that's","it's","your"
#         ])
#         self.lemmatizer = WordNetLemmatizer()

#     def parse(self, ngrams=1):
#         words = re.findall(r'\b\w+\b', self.sentence.lower())
#         words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
#         if ngrams > 1:
#             words = self._make_ngrams(words, ngrams)
#         return words

#     def _make_ngrams(self, words, n):
#         return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# # ==========================
# # BAYES CLASSIFIER TF-IDF + DICTIONARY
# # ==========================
# class Bayes_Classifier:
#     def __init__(self):
#         self.pos_dic = defaultdict(float)
#         self.neg_dic = defaultdict(float)
#         self.vocab = set()
#         self.df = defaultdict(int)
#         self.doc_count = 0
#         self.pos_docs = 0
#         self.neg_docs = 0
#         self.pos_word_sum = 0
#         self.neg_word_sum = 0
#         self.prob_pos = 0.5
#         self.prob_neg = 0.5

#         # Dictionary cảm xúc
#         self.emotion_dict = {
#             "love": "5",
#             "happy": "5",
#             "amazing": "5",
#             "fantastic": "5",
#             "great": "5",
#             "good": "5",
#             "excellent": "5",
#             "hate": "1",
#             "sad": "1",
#             "terrible": "1",
#             "bad": "1",
#             "awful": "1",
#             "horrible": "1"
#         }

#     # ------------------------
#     # TRAINING
#     # ------------------------
#     def train(self, lines):
#         docs_words = []
#         total_valid = 0

#         for line in lines:
#             if not line or "|" not in line:
#                 continue
#             parts = line.strip().split("|", 2)
#             if len(parts) < 3:
#                 continue
#             label, _, text = parts
#             words = SentenceParser(text).parse(ngrams=2)  # dùng bigram
#             if not words:
#                 continue

#             docs_words.append((label, words))
#             total_valid += 1

#             unique_words = set(words)
#             for w in unique_words:
#                 self.df[w] += 1

#             if label == "5":
#                 self.pos_docs += 1
#             else:
#                 self.neg_docs += 1

#         self.doc_count = total_valid
#         self.prob_pos = self.pos_docs / total_valid if total_valid > 0 else 0.5
#         self.prob_neg = self.neg_docs / total_valid if total_valid > 0 else 0.5

#         N = self.doc_count
#         for label, words in docs_words:
#             tf = Counter(words)
#             for w, count in tf.items():
#                 idf = math.log((N + 1) / (self.df[w] + 1)) + 1
#                 tf_idf = count * idf
#                 self.vocab.add(w)
#                 if label == "5":
#                     self.pos_dic[w] += tf_idf
#                     self.pos_word_sum += tf_idf
#                 else:
#                     self.neg_dic[w] += tf_idf
#                     self.neg_word_sum += tf_idf

#         self.total_words = len(self.vocab) or 1

#     # ------------------------
#     # SCORE
#     # ------------------------
#     def _score(self, words, label):
#         s = math.log(self.prob_pos if label=="5" else self.prob_neg)
#         dic = self.pos_dic if label=="5" else self.neg_dic
#         total_tf_idf = self.pos_word_sum if label=="5" else self.neg_word_sum
#         V = self.total_words

#         for w in words:
#             val = dic.get(w, 0) + 1
#             s += math.log(val / (total_tf_idf + V))
#         return s

#     # ------------------------
#     # CLASSIFY SINGLE TEXT
#     # ------------------------
#     def classify_text(self, text):
#         words = SentenceParser(text).parse(ngrams=1)

#         # Kiểm tra dictionary
#         for w in words:
#             if w in self.emotion_dict:
#                 label = self.emotion_dict[w]
#                 prob_pos = 1.0 if label=="5" else 0.0
#                 prob_neg = 1.0 - prob_pos
#                 return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

#         # Nếu không có trong dictionary, dùng Bayes TF-IDF
#         p_score = self._score(words, "5")
#         n_score = self._score(words, "1")
#         label = "5" if p_score >= n_score else "1"

#         try:
#             prob_pos = 1 / (1 + math.exp(n_score - p_score))
#             prob_neg = 1 - prob_pos
#         except:
#             prob_pos, prob_neg = 0.5, 0.5

#         return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

# # ==========================
# # TEST NHANH
# # ==========================
# if __name__ == "__main__":
#     training_data = [
#         "5|0|I love this product",
#         "5|0|This is amazing",
#         "5|0|Fantastic work",
#         "5|0|happy",
#         "1|0|I hate this",
#         "1|0|Very sad experience",
#         "1|0|Terrible outcome",
#         "1|0|sad"
#     ]

#     classifier = Bayes_Classifier()
#     classifier.train(training_data)

#     # Test nhiều từ/câu ngắn
#     test_texts = ["sad", "I am very sad today", "fantastic", "I hate it", "love", "happy", "terrible"]
#     for t in test_texts:
#         res = classifier.classify_text(t)
#         print(f"Input: {t}\nOutput: {res}\n")
import math
import re
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter

# Tải dữ liệu NLTK cần thiết
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==========================
# SENTENCE PARSER
# ==========================
class SentenceParser:
    def __init__(self, sentence):
        self.sentence = sentence or ""
        self.stop_words = set([
            "", "i","me","my","myself","you","your","he","him","his",
            "she","her","it","its","we","us","they","them","this","that",
            "those","with","am","is","are","was","were","to","a","an","the",
            "of","for","in","on","at","and","or","as","by","do","doing","from",
            "have","has","had","be","been","being","there","here","when","where",
            "who","which","because","due","after","before","let","make","get",
            "out","still","that's","it's","your"
        ])
        self.lemmatizer = WordNetLemmatizer()

    def parse(self, ngrams=1):
        words = re.findall(r'\b\w+\b', self.sentence.lower())
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        if ngrams > 1:
            words = self._make_ngrams(words, ngrams)
        return words

    def _make_ngrams(self, words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# ==========================
# BAYES CLASSIFIER TF-IDF + DICTIONARY
# ==========================
class Bayes_Classifier:
    def __init__(self):
        self.pos_dic = defaultdict(float)
        self.neg_dic = defaultdict(float)
        self.vocab = set()
        self.df = defaultdict(int)
        self.doc_count = 0
        self.pos_docs = 0
        self.neg_docs = 0
        self.pos_word_sum = 0
        self.neg_word_sum = 0
        self.prob_pos = 0.5
        self.prob_neg = 0.5

        # Dictionary cảm xúc
        self.emotion_dict = {
            "love": "5",
            "happy": "5",
            "amazing": "5",
            "fantastic": "5",
            "great": "5",
            "good": "5",
            "excellent": "5",
            "hate": "1",
            "sad": "1",
            "terrible": "1",
            "bad": "1",
            "awful": "1",
            "horrible": "1"
        }

    # ------------------------
    # TRAINING
    # ------------------------
    def train(self, lines):
        docs_words = []
        total_valid = 0

        for line in lines:
            if not line or "|" not in line:
                continue
            parts = line.strip().split("|", 2)
            if len(parts) < 3:
                continue
            label, _, text = parts
            words = SentenceParser(text).parse(ngrams=2)  # dùng bigram
            if not words:
                continue

            docs_words.append((label, words))
            total_valid += 1

            unique_words = set(words)
            for w in unique_words:
                self.df[w] += 1

            if label == "5":
                self.pos_docs += 1
            else:
                self.neg_docs += 1

        self.doc_count = total_valid
        self.prob_pos = self.pos_docs / total_valid if total_valid > 0 else 0.5
        self.prob_neg = self.neg_docs / total_valid if total_valid > 0 else 0.5

        N = self.doc_count
        for label, words in docs_words:
            tf = Counter(words)
            for w, count in tf.items():
                idf = math.log((N + 1) / (self.df[w] + 1)) + 1
                tf_idf = count * idf
                self.vocab.add(w)
                if label == "5":
                    self.pos_dic[w] += tf_idf
                    self.pos_word_sum += tf_idf
                else:
                    self.neg_dic[w] += tf_idf
                    self.neg_word_sum += tf_idf

        self.total_words = len(self.vocab) or 1

    # ------------------------
    # SCORE
    # ------------------------
    def _score(self, words, label):
        s = math.log(self.prob_pos if label=="5" else self.prob_neg)
        dic = self.pos_dic if label=="5" else self.neg_dic
        total_tf_idf = self.pos_word_sum if label=="5" else self.neg_word_sum
        V = self.total_words

        for w in words:
            val = dic.get(w, 0) + 1
            s += math.log(val / (total_tf_idf + V))
        return s

    # ------------------------
    # CLASSIFY SINGLE TEXT
    # ------------------------
    def classify_text(self, text):
        words = SentenceParser(text).parse(ngrams=1)

        # Kiểm tra dictionary trước
        for w in words:
            if w in self.emotion_dict:
                label = self.emotion_dict[w]
                prob_pos = 1.0 if label=="5" else 0.0
                prob_neg = 1.0 - prob_pos
                return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

        # Nếu không có trong dictionary, dùng Bayes TF-IDF
        p_score = self._score(words, "5")
        n_score = self._score(words, "1")
        label = "5" if p_score >= n_score else "1"

        try:
            prob_pos = 1 / (1 + math.exp(n_score - p_score))
            prob_neg = 1 - prob_pos
        except:
            prob_pos, prob_neg = 0.5, 0.5

        return {"label": label, "score_pos": prob_pos, "score_neg": prob_neg, "status":"ok"}

    # ------------------------
    # CLASSIFY MULTIPLE LINES
    # ------------------------
    def classify_lines(self, lines):
        results = []
        for line in lines:
            # Nếu dòng là dataset dạng label|id|text
            if "|" in line:
                parts = line.strip().split("|", 2)
                if len(parts) == 3:
                    _, _, text = parts
                else:
                    text = line
            else:
                text = line
            res = self.classify_text(text)
            results.append(res["label"])
        return results

# ==========================
# TEST NHANH
# ==========================
if __name__ == "__main__":
    training_data = [
        "5|0|I love this product",
        "5|0|This is amazing",
        "5|0|Fantastic work",
        "5|0|happy",
        "1|0|I hate this",
        "1|0|Very sad experience",
        "1|0|Terrible outcome",
        "1|0|sad"
    ]

    classifier = Bayes_Classifier()
    classifier.train(training_data)

    # Test nhiều từ/câu ngắn
    test_texts = ["sad", "I am very sad today", "fantastic", "I hate it", "love", "happy", "terrible"]
    for t in test_texts:
        res = classifier.classify_text(t)
        print(f"Input: {t}\nOutput: {res}\n")
