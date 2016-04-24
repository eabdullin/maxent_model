# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import random
import re


def predict(x, weights, y_patterns):
    # начальное приведение
    probas = np.ones(weights.shape[1]) * np.log(1.0 / weights.shape[1])

    # считаем условные вероятности
    for xi in x:
        v = weights[xi] * y_patterns[xi]
        probas += v

    # далее сглаживаем выходы через softmax
    probas = np.exp(probas / weights.shape[1])
    return probas / np.sum(probas)


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


def fit(X, y, alpha=0.85, max_iter=100, tol=0.00001, random_state=None, verbose=1):
    n_samples = len(X)
    if random_state is not None:
        random.seed(random_state)

    # определяем сколько у нас уникальных токенов
    features = unique(X)

    # определяем сколько у нас уникальных классов
    classes = unique(y)

    # матрица индикаторов(условных признаков)
    feature_patterns = np.zeros((features.shape[0], classes.shape[0]), dtype=np.int)

    # матрица весов индикаторов
    weights = np.zeros((features.shape[0], classes.shape[0]))

    # инициализация индикаторов
    for i in range(n_samples):
        for xi in X[i]:
            feature_patterns[xi, y[i]] = 1

    #
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

            if verbose and iter_sample % (n_samples / 20) == 0:
                print '.',

            all_iter += 1
            eta = alpha ** (all_iter / n_samples)
            # предсказываем вероятности
            probas = predict(X[i], weights, feature_patterns)

            # смотрим, правильно ли мы предсказали, это нужно только для verbose
            if np.argmax(probas) == y[i]:
                ncorrect += 1
            # считаем "правдоподобие"
            logl += np.log(probas[y[i]]) / features.shape[0]

            # обновляем веса
            for j in range(len(X[i])):
                conditional_y = feature_patterns[X[i][j]]
                for y_i in range(len(conditional_y)):
                    # ожидание
                    expected_ent = 1.0 if conditional_y[y_i] == 1 and y_i == y[i] else 0.0
                    # реальность
                    max_ent = probas[y_i]
                    weights[X[i][j], y_i] -= (max_ent - expected_ent) * eta  #

        if iter_num > 0:
            if prev_logl > logl:
                print('there is model diverging')
                break
            if (logl - prev_logl) < tol:
                break

        if verbose:
            print '\tAccuracy: %.5f, Loss: %.8f' % (1.0 * ncorrect / n_samples, logl - prev_logl)
        prev_logl = logl
    print iter_num
    return weights, feature_patterns


# X = [[0, 1],
#      [2, 1],
#      [2, 3],
#      [2, 1],
#      [0, 1],
#      [2, 1, 4],
#      [2, 3, 4],
#      [2, 1, 5],
#      [0, 3, 5],
#      [0, 1, 5]]
#
# y = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0]
#
# weights, patterns = fit(X, y)
# print weights
# print patterns
#
# pred = predict([0, 1, 2, 2, 2], weights, patterns)
# print pred

digits_regex = re.compile('\d')
punc_regex = re.compile('[\%\(\)\-\/\:\;\<\>\«\»]')
delim_regex = re.compile('([\.\,])\s+')
word2index = {'.': 0}
index2word = ['.']
i = 1
tokenized_text = []
for path, subdirs, files in os.walk('data'):
    for name in files:
        filename = os.path.join(path, name)
        with io.open(filename, 'r', encoding='utf-8') as data_file:
            text = digits_regex.sub(u'0', data_file.read().lower())
            text = punc_regex.sub(u'', text)
            text = delim_regex.sub(r' \1 ', text)
            for word in text.split():
                if word and word not in word2index:
                    word2index[word] = i
                    index2word.append(word)
                    i += 1
                tokenized_text.append(word)
print len(tokenized_text)
# for w in tokenized_text[:30]:
#     print w,
X = []
y = []
for i, y_word in enumerate(tokenized_text):
    x = []
    for j in range(i - 5, i):
        if (j >= 0):
            x_word = tokenized_text[j]
            x.append(word2index[x_word])
    if (len(x) > 0):
        X.append(x)
        y.append(word2index[y_word])

weights, patterns = fit(X, y)
print weights
print patterns

pred = predict([1, 2, 3, 4], weights, patterns)
index = np.argmax(pred)
print index2word[index]
