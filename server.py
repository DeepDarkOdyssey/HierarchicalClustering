# -*- coding: utf-8 -*-

import json
from flask import Flask, request
from hierarchical_clustering import HierarchicalClustering
from distance_measures import lcs_distance
from utils import preprocess, load_stopwords


app = Flask(__name__)
stopwords = load_stopwords('stopwords.txt')
models = {}


@app.route('/')
def hello():
    return 'hello'


@app.route('/create/<model_name>')
def create(model_name):
    app.logger.info('model {} has been created'.format(model_name))
    model = HierarchicalClustering(model_name, lcs_distance, 0.7, True)
    models[model_name] = model
    return 'creation finished'


@app.route('/clear/<model_name>')
def clear(model_name):
    model = models[model_name]
    model.clear()
    app.logger.info('bomb {} has been cleared'.format(model_name))
    return 'clear'


@app.route('/update/<model_name>', methods=['POST'])
def update(model_name):
    if model_name not in models:
        create(model_name)

    raw_samples = json.loads(request.json)
    samples = preprocess(raw_samples, True, stopwords)

    model = models[model_name]
    changed_samples = model.fit(samples)

    return json.dumps(changed_samples)


@app.route('/remove/<model_name>', methods=['POST'])
def remove(model_name):
    if model_name not in models:
        return 'this model has not been created'
    sample_ids = json.loads(request.json)
    model = models[model_name]

    model.drop_samples(sample_ids)
    app.logger.info('{} has been removed from model {}'.format(sample_ids, model_name))

    return 'removed'


@app.route('/showall')
def showall():
    for hc in models.values():
        print('model name :{}'.format(hc.name))
        for cluster_id, sample_ids in hc.clusters.items():
            print('cluster {}'.format(cluster_id))
            for sample_id in sample_ids:
                print(sample_id, hc.samples[sample_id].title)
        print()
    return 'Done'


if __name__ == '__main__':
    app.run(port=10000, debug=True)
