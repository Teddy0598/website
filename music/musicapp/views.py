from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
# Create your views here.


def home(request):
    return render(request, 'index.html')


def external(request):

    audio = request.FILES['audio']
    print("Audio :", audio)
    fs = FileSystemStorage()
    filename = fs.save(audio.name, audio)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print('file raw url', filename)
    print("file full url", fileurl)
    print("template url", templateurl)

    def distance(instance1, instance2, k):
        distance = 0
        mm1 = instance1[0]
        cm1 = instance1[1]
        mm2 = instance2[0]
        cm2 = instance2[1]
        distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        distance -= k
        return distance

    def Neighbors(training_set, instance, k):
        distances = []
        for x in range(len(training_set)):
            dist = distance(training_set[x], instance, k) + distance(instance, training_set[x], k)
            distances.append((training_set[x][2], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors

    def nearestClass(neighbors):
        vote = {}

        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in vote:
                vote[response] += 1
            else:
                vote[response] = 1

        sorter = sorted(vote.items(), key=operator.itemgetter(1), reverse=True)

        return sorter[0][0]

    def Accuracy(test_set, predictions):
        correct = 0
        for X in range(len(test_set)):
            if test_set[X][-1] == predictions[X]:
                correct += 1

        return (1.0 * correct) / len(test_set)

    directory = "./genres/"

    dataset = []

    def HandlingDataset(filename, split, trSet, teSet):
        with open('my.dat', 'rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break
        for x in range(len(dataset)):
            if random.random() < split:
                trSet.append(dataset[x])
            else:
                teSet.append(dataset[x])

    training_set = []
    test_set = []
    HandlingDataset('my.dat', 0.66, training_set, test_set)

    leng = len(test_set)
    predictions = []
    for x in range(leng):
        predictions.append(nearestClass(Neighbors(training_set, test_set[x], 4)))

    accuracy1 = Accuracy(test_set, predictions)
    res = accuracy1 * 100
    print(f"Accuracy: {round(res, 2)}%")

    from collections import defaultdict

    results = defaultdict(int)
    # directory = './genres/'
    i = 1
    for folder in os.listdir(directory):
        results[i] = folder
        i += 1

    # test_dir = fs.open(filename)
    # test_name = fs.save(audio.name, audio)
    test_file = fs.open(filename)
    # test_file = test_dir + "test2.wav"
    # test_file = test_dir + "test3.wav"
    # test_file = test_dir + "test4.wav"

    (rate, sig) = wav.read(test_file)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, i)

    pred = nearestClass(Neighbors(dataset, feature, 4))
    print(f"Your music Genre is {results[pred]}")
    data1 = round(res, 2)
    data2 = results[pred]
    print(data1)
    print(data2)

    return render(request, 'index.html', {'data1': data1, 'data2': data2})
