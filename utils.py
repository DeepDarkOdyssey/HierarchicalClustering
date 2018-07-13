# -*- coding: utf-8 -*-

import re
import json
import urllib
from string import punctuation
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
    """输入原始的samples，将其进行预处理、分词后返回

    Args:
        raw_samples: list of dict, 每个dict为一个原始的sample，每个sample至少包含两个key: 'id'和'title'
        segmentation: bool，是否分词
        stopwords: list of str, 停用词表

    Returns:
        处理后的samples, 每个sample添加了额外的'seg_title', 'cluster_id' 和 'distance2samples'的key
    """
    samples = []
    for raw_sample in raw_samples:
        i = raw_sample['id']
        raw_title = raw_sample['title']
        # table = str.maketrans({key: None for key in punctuation})
        # title = title.translate(table)
        # title = re.sub('\d|\ufeff', '', title)
        title = remove_punctuation(raw_title)
        if segmentation:
            segmented_title = word_segment_own(title)
            segmented_title = [token for token in segmented_title if token not in stopwords]
        else:
            segmented_title = []
        sample_dict = {'id': i, 'raw_title': raw_title, 'title': title, 'seg_title': segmented_title,
                       'cluster_id': None, 'distance2samples': {}}
        sample = Munch(sample_dict)
        samples.append(sample)

    return samples


def load_stopwords(file_path):
    """从指定文件路径读取停用词"""

    stopwords = []
    with open(file_path) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            stopwords.append(line.strip())
    stopwords.extend(['年', '月', '日', '月日', '年月日', ' '])
    return stopwords


def remove_punctuation(string):
    """去除所有的标点等特殊符号，只保留输入的str中的英文大小写字母，中文字符并返回"""

    pattern = re.compile(r"[^a-zA-Z\u4e00-\u9fa5]")
    string = re.sub(pattern, '', string)
    return string
