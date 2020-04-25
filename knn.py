import numpy as np
import math
import sys
import re
from flask import Flask, render_template, redirect, request, session, url_for, g


class Knn:
    def __init__(self, datafile, labelfile):
        data = self.parsepoints(datafile)
        labels = self.parselabels(labelfile)
        zipped = list(zip(data, labels))
        self.datapoints = {x[0]: x[1] for x in zipped}
        print(self.datapoints)

    @staticmethod
    def distance(x, y):
        summation = 0
        xx = np.array(x)
        yy = np.array(y)
        for i in range(xx.shape[0]):
            summation += (yy[i] - xx[i]) ** 2
        return math.sqrt(summation)

    @staticmethod
    def parsepoints(datafile):
        data = []
        with open(datafile, 'r') as fp:
            for l in fp:
                point = []
                spl = l.split(",")
                for s in spl:
                    point.append(float(s))
                t = tuple(point)
                data.append(t)
        return data

    @staticmethod
    def parselabels(labelfile):
        labels = []
        with open(labelfile) as fp:
            for l in fp:
                labels.append(int(l.split("\n")[0]))
        return labels

    def classify(self, point, k):
        distances = []
        neighbors = []
        for ke in self.datapoints.keys():
            d = self.distance(point, ke)
            distances.append((ke, d))

        for i in range(k):
            minimum = min(distances, key=lambda x: x[1])
            neighbors.append((minimum[0], self.datapoints[minimum[0]]))
            distances.remove(minimum)
            print(distances)
            print(neighbors)

        if k is 1:
            return neighbors[0][1]
        else:
            freq = {}
            for x in neighbors:
                freq[x[1]] = freq.get(x[1], 0) + 1
            return max(freq, key=freq.get)

        pass


app = Flask(__name__)
app.config.from_json('app.cfg', silent=True)


@app.route('/')
def root():
    return render_template("index.html")


@app.route('/knn')
def knearestneighbors():
    return render_template("knn.html")


@app.route('/uploader', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        d = request.files['data']
        l = request.files['labels']
        d.save("data.txt")
        l.save("l.txt")
        session['data'] = "data.txt"
        session['label'] = "l.txt"
        return redirect(url_for('predict'))


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        result = request.form
        m = Knn(session['data'], session['label'])
        s = re.sub(r'\s+', '', result["point"])
        print(s)
        sp = s.split(",")
        point = []
        for i in sp:
            point.append(float(i))
        classification = m.classify(point, int(result["k"]))
        return render_template("knnResult.html", result=classification)
    if request.method == 'GET':
        return render_template("knnPredict.html")


if __name__ == '__main__':
    app.run()
