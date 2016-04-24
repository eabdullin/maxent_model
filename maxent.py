# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import random
import re

def unique(arr, dic=None):
    if (dic is None):
        dic = {}
    for el in arr:
        if isinstance(el, list):
            unique(el, dic)
        else:
            if (el not in dic):
                dic[el] = 1
            else:
                dic[el] += 1
    return np.array(dic.keys())

def predict(x, weights, y_patterns, n_classes):
    # начальное приведение
    probas = np.ones(n_classes) * np.log(1.0 / n_classes)

    # считаем условные вероятности
    for xi in x:
        for i, yi in enumerate(y_patterns[xi]):
            probas[yi] += weights[xi][i]

    # далее сглаживаем выходы через softmax
    probas = np.exp(probas / n_classes)
    return probas / np.sum(probas)

def fit(X, y, f_count, c_count, alpha=0.85, max_iter=100, tol=0.00001, random_state=None, verbose=1):
    n_samples = len(X)
    if random_state is not None:
        random.seed(random_state)

#     # определяем сколько у нас уникальных токенов
#     features = unique(X)
#     f_count = features.shape[0]
#     # определяем сколько у нас уникальных классов
#     classes = unique(y)
#     c_count = classes.shape[0]

    # матрица индикаторов(условных признаков)
    feature_patterns = [[] for _ in range(f_count)]
    f_pattern_set = {}
    # матрица весов индикаторов
    weights = [[] for _ in range(f_count)]

    # инициализация индикаторов
    for i in range(n_samples):
        for xi in X[i]:
            if xi not in f_pattern_set:
                f_pattern_set[xi] = set()
            if y[i] not in f_pattern_set[xi]:
                f_pattern_set[xi].add(y[i])
                weights[xi].append(0.0)
                feature_patterns[xi].append(y[i])
    print feature_patterns[:10]
    prev_logl = 0.
    iter_num = 0
    all_iter = 0
    # ограничим сверху max_iter итерациями
    for iter_num in range(max_iter):
        if verbose:
            print 'Start iteration #%d\t' % iter_num,

        logl = 0.
        ncorrect = 0

        # random прохождение существенно улучшает схождение SGD
        r = range(n_samples)
        r = random.sample(r, n_samples)
        iter_sample = 0
        for i in r:
            iter_sample += 1
            if verbose and (n_samples < 20 or iter_sample % (n_samples / 20) == 0):
                print '.',

            all_iter += 1
            eta = alpha ** (all_iter / n_samples)
            # предсказываем вероятности
            probas = predict(X[i], weights, feature_patterns, c_count)

            # смотрим, правильно ли мы предсказали, это нужно только для verbose
            if np.argmax(probas) == y[i]:
                ncorrect += 1
            # считаем "правдоподобие"
            logl += np.log(probas[y[i]])

            # обновляем веса
            for j in range(len(X[i])):
                for y_i, con_y in enumerate(feature_patterns[X[i][j]]):
                    # ожидание
                    expected_ent = 1.0 if con_y == y[i] else 0.0
                    # реальность
                    max_ent = probas[con_y]
                    weights[X[i][j]][y_i] -= 1.0 * (max_ent - expected_ent) * eta  #
        logl /= n_samples
        if verbose:
            print '\tAccuracy: %.5f, Loss: %.8f' % (1.0 * ncorrect / n_samples, logl - prev_logl)
        if iter_num > 0:
            if prev_logl > logl:
                break
            if (logl - prev_logl) < tol:
                break
        prev_logl = logl
    print iter_num
    return weights, feature_patterns

digits_regex = re.compile('\d')
punc_regex = re.compile('[\%\(\)\-\/\:\;\<\>\«\»\,]')
delim_regex = re.compile('([\.])\s+')

def read_and_tokenize(foldername):
    '''
    метод для считывания текстов из файлов папки
    здесь применяется довольно простая токенизация
    '''

    word_counts = {}
    tokenized_text = []
    for path, subdirs, files in os.walk('data'):
        for name in files:
            filename = os.path.join(path, name)
            with io.open(filename, 'r', encoding='utf-8') as data_file:
                for line in data_file:
                    if len(line) < 50:
                        continue
                    text = digits_regex.sub(u'0', line.lower())
                    text = punc_regex.sub(u'', text)
                    text = delim_regex.sub(r' \1 ', text)
                    for word in text.split():
                        if not word:
                            continue
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1
                        tokenized_text.append(word)
    word2index = {}
    index2word = []
    i = 0
    filtered_text = []
    for word in tokenized_text:
        if word_counts[word] > 10:
            if word not in word2index:
                word2index[word] = i
                index2word.append(word)
                i += 1
            filtered_text.append(word)


    return filtered_text, word2index, index2word

def generate_train(tokenized_text, word2index,context_len = 4):
    '''
    метод для генерации обучающих данных
    '''
    X = []
    y = []
    for i, y_word in enumerate(tokenized_text):
        x = []
        for j in range(i - context_len, i):
            if (j >= 0):
                x_word = tokenized_text[j]
                x.append(word2index[x_word])
        if (len(x) > 0):
            X.append(x)
            y.append(word2index[y_word])
    return X, y

tokenized_text, word2index, index2word = read_and_tokenize('data')

unique_words = len(index2word)
print 'all words:', len(tokenized_text)
print 'all unique words', unique_words

context_len = 4
X,y = generate_train(tokenized_text, word2index,context_len=context_len)

weights, patterns = fit(X, y,unique_words,unique_words,random_state=241,verbose=1)

test = [word2index[u'экономическая'],word2index[u'ситуация']]
for i in range(10):
    pred = predict(test, weights, patterns)
    index = np.argmax(pred)
    print index2word[index],
    test.append(index)
    if len(test) > context_len:
        del test[0]
    print test