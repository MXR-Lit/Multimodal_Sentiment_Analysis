import os
import re
from gensim.models import Word2Vec

def make_word2vec():
    max_lenth=0
    corpus = []
    folder_path = './data'
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read().strip()
                cleaned_text = re.sub(r'[^\w\s]', '', text)  # 移除非单词字符和空白字符
                lowercase_text = cleaned_text.lower()  # 将文本转换为小写字母
                text = lowercase_text.split()
                corpus.append(text)
                if len(text)>max_lenth:
                    max_lenth=len(text)
            f.close()
    # print(corpus)
    #print(max_lenth)

    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=1, epochs=100)
    model.save("word2vec_model.bin")
    #word_vectors = model.wv
    #print(word_vectors['forest'])
    #model.train(corpus,total_examples=model.corpus_count,epochs=model.epochs+10)


if __name__ == '__main__':
    make_word2vec()
    load_model=Word2Vec.load("word2vec_model.bin")
    wordvectors=load_model.wv
    print(wordvectors['forest'])
