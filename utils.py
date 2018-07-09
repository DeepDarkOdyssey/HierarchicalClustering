# -*- coding: utf-8 -*-

import re
import json
import urllib
from munch import Munch


def get_title_content(fpath):
    with open(fpath, encoding='utf8') as f:
        title = f.readline()
        content = f.read()
        segments = re.findall(r'>(.*?)<', content)
        content = ''
        for segment in segments:
            segment = re.sub('&gt;', '', segment)
            content += segment
    return title.strip(), content


def word_segment_own(string):
    url_get_base = "http://10.200.7.53:7022/wordsegment/segment?"
    args = {
            'analyzer': '4',
            'words': string,
            'resulttype': '1',
            'segmenttype': '1',
            'encode': 'utf-8',
            'combine': '0'
        }
    result = urllib.request.urlopen(url_get_base, data=urllib.parse.urlencode(args).encode(encoding='UTF8')) # POST method
    content = result.read().strip()
    dic = json.loads(content.decode('utf-8'))
    wordList = dic.get('wordList')
    segresult = []
    for entry in wordList:
        segresult.append(entry.get("word"))
    return segresult


def preprocess(raw_samples, segmentation=False, stopwords=[]):
    samples = []
    for raw_sample in raw_samples:
        i = raw_sample['id']
        title = raw_sample['title']
        title = re.sub('\d|\ufeff', '', title)
        if segmentation:
            segmented_title = word_segment_own(title)
            segmented_title = [token for token in segmented_title if token not in stopwords]
        else:
            segmented_title = []
        sample_dict = {'id': i, 'title': title, 'seg_title': segmented_title, 'cluster_id': None, 'distance2samples': {}}
        sample = Munch(sample_dict)
        samples.append(sample)

    return samples


def load_stopwords(file_path):
    stopwords = []
    with open(file_path) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            stopwords.append(line.strip())
    stopwords.extend(['年', '月', '日', ' '])
    return stopwords

