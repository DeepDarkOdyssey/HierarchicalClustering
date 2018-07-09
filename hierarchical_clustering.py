# -*- coding: utf-8 -*-

from collections import defaultdict
import re
import os
import time
import pickle
import logging
from utils import word_segment_own, get_title_content, preprocess
from distance_measures import jaccard_distance, lcs_distance
from munch import Munch


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')


class HierarchicalClustering:
    def __init__(self, name, dist_func, threshold, use_buckets=False, close_num=10):
        self.name = name
        self.dist_func = dist_func
        self.threshold = threshold
        self.use_buckets = use_buckets
        self.close_num = close_num

        self._clusters = {}
        self.samples = {}
        self.sample_ids = []
        self._current_cluster_id = -1
        self.bucket2id = defaultdict(list)
        self.id2bucket = defaultdict(list)
        self._changed_sample_ids = set()

        self.computation_count = 0
        self.mapping_count = 0
        self.loop_count = 0

    @property
    def clusters(self):
        clusters = defaultdict(list)
        for cluster_id, cluster in self._clusters.items():
            sample_ids = cluster.sample_ids
            if len(sample_ids) < 2:
                clusters[-1].extend(sample_ids)
            else:
                if -1 in clusters:
                    clusters[len(clusters)-1] = sample_ids
                else:
                    clusters[len(clusters)] = sample_ids
        return clusters

    @property
    def current_cluster_id(self):
        self._current_cluster_id += 1
        return self._current_cluster_id

    @property
    def changed_samples(self):
        changed_samples = {}
        for sample_id in self._changed_sample_ids:
            changed_samples[sample_id] = self.samples[sample_id].cluster_id
        self._changed_sample_ids.clear()
        return changed_samples

    def clear(self):
        self._clusters = {}
        self.samples = {}
        self.sample_ids = []
        self._current_cluster_id = -1
        self.bucket2id = defaultdict(list)
        self.id2bucket = defaultdict(list)
        self._changed_sample_ids = set()

        self.computation_count = 0
        self.mapping_count = 0
        self.loop_count = 0

    def save_to(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(hc, f)

    def load_from(self, file_path):
        with open(file_path, 'rb') as f:
            tmp_model = pickle.load(f)
        self._clusters = tmp_model._clusters
        self.samples = tmp_model.samples
        self.sample_ids = tmp_model.sample_ids
        self._current_cluster_id = -1
        self.bucket2id = tmp_model.bucket2id
        self.id2bucket = tmp_model.id2bucket
        self._changed_sample_ids = set()
        del tmp_model

    def drop_samples(self, sample_ids):
        for sample_id in sample_ids:
            correlated_sample = self.samples[sample_id]
            self.sample_ids.remove(sample_id)
            self.samples.pop(sample_id)
            for sample in self.samples:
                sample.distance2samples.pop(sample_id)

            correlated_cluster = self._clusters[correlated_sample.cluster_id]
            correlated_cluster.sample_ids.remove(sample_id)
            for cluster_id, cluster in self._clusters.items():
                if cluster_id != correlated_cluster.id:
                    self.compute_cluster_distance(correlated_cluster, cluster)

    def add_samples(self, samples):
        for sample in samples:
            if sample.id in self.sample_ids:
                raise IndexError('The sample {} has already been added to the model'.format(sample.id))
            else:
                self.sample_ids.append(sample.id)
                self._changed_sample_ids.add(sample.id)
                self.samples[sample.id] = sample
                if sample.cluster_id is None:
                    cluster_id = self.current_cluster_id
                    sample.cluster_id = cluster_id
                if sample.cluster_id not in self._clusters:
                    cluster_dict = {'id': sample.cluster_id, 'sample_ids': [sample.id], 'distance2clusters': {}}
                    self._clusters[sample.cluster_id] = Munch(cluster_dict)
                if self.use_buckets and sample.seg_title:
                    for token in sample.seg_title:
                        self.bucket2id[token].append(sample.id)
                        self.id2bucket[sample.id].append(token)

    def compute_sample_distance(self, s1, s2):
        if s2.id in s1.distance2samples:
            dist = s1.distance2samples[s2.id]
            self.mapping_count += 1

        elif s1.id in s2.distance2samples:
            dist = s2.distance2samples[s2.id]
            self.mapping_count += 1

        else:
            if self.use_buckets and len(self.bucket2id) > 0:
                in_same_bucket = False
                dist = 1
                for bucket in self.id2bucket[s1.id]:
                    if s2.id in self.bucket2id[bucket]:
                        in_same_bucket = True
                        break
                if in_same_bucket:
                    dist = self.dist_func([ord(c) for c in s1.title], [ord(c) for c in s2.title])
                    # dist = self.dist_func(s1.title, s2.title)
            else:
                dist = self.dist_func([ord(c) for c in s1.title], [ord(c) for c in s2.title])
                # dist = self.dist_func(s1.title, s2.title)
            self.computation_count += 1
            s1.distance2samples[s2.id] = dist
            s2.distance2samples[s1.id] = dist
        return dist

    def compute_cluster_distance(self, c1, c2):
        if c2.id in c1.distance2clusters:
            dist = c1.distance2clusters[c2.id]
            self.mapping_count += 1

        elif c1.id in c2.distance2clusters:
            dist = c2.distance2clusters[c1.id]
            self.mapping_count += 1

        else:
            distances = []
            for s1_id in c1.sample_ids:
                for s2_id in c2.sample_ids:
                    distance = self.compute_sample_distance(self.samples[s1_id], self.samples[s2_id])
                    distances.append(distance)
            dist = max(distances)
            c1.distance2clusters[c2.id] = dist
            c2.distance2clusters[c1.id] = dist
        return dist

    def compute_cluster_distance_prev(self, c1, c2):
        distances = []
        for s1_id in c1.sample_ids:
            for s2_id in c2.sample_ids:
                distance = self.compute_sample_distance(self.samples[s1_id], self.samples[s2_id])
                distances.append(distance)
        dist = max(distances)
        c1.distance2clusters[c2.id] = dist
        c2.distance2clusters[c1.id] = dist
        return dist

    def find_nearest_cluster_pair(self):
        min_dist = 100
        cluster_pair = None
        clusters = self._clusters.copy()
        while True:
            try:
                cluster_id, cluster = clusters.popitem()
            except KeyError:
                break
            for another_id, another_cluster in clusters.items():
                if another_id != cluster_id:
                    self.loop_count += 1
                    dist = self.compute_cluster_distance(cluster, another_cluster)
                    if dist < min_dist and dist <= self.threshold:
                        cluster_pair = (cluster_id, another_id)
                        min_dist = dist
        return cluster_pair

    def find_close_cluster_pairs(self):
        close_cluster_pairs = []
        clusters = self._clusters.copy()
        while True:
            try:
                cluster_id, cluster = clusters.popitem()
            except KeyError:
                break
            for another_id, another_cluster in clusters.items():
                self.loop_count += 1
                dist = self.compute_cluster_distance(cluster, another_cluster)
                if dist <= self.threshold:
                    close_cluster_pairs.append((dist, (cluster_id, another_id)))

        cluster_pairs = []
        added_cluster_ids = []
        for dist, cluster_pair in sorted(close_cluster_pairs):
            if cluster_pair[0] not in added_cluster_ids and cluster_pair[1] not in added_cluster_ids:
                cluster_pairs.append(cluster_pair)
                added_cluster_ids.extend(cluster_pair)
        return cluster_pairs[:self.close_num]

    def merge_cluster_pair(self, c1, c2):
        new_cluster_id = self.current_cluster_id
        new_cluster = Munch(id=new_cluster_id, sample_ids=[], distance2clusters={})
        s1_ids = c1.sample_ids
        s2_ids = c2.sample_ids
        for i in s1_ids:
            self.samples[i].cluster_id = new_cluster_id
            self._changed_sample_ids.add(i)
        for i in s2_ids:
            self.samples[i].cluster_id = new_cluster_id
            self._changed_sample_ids.add(i)
        new_cluster.sample_ids.extend(s1_ids)
        new_cluster.sample_ids.extend(s2_ids)
        for cluster_id, distance in c1.distance2clusters.items():
            if cluster_id != c1.id and cluster_id != c2.id:
                dist = max(distance, c2.distance2clusters[cluster_id])
                new_cluster.distance2clusters[cluster_id] = dist
        del self._clusters[c1.id]
        del self._clusters[c2.id]
        for c in self._clusters.values():
            del c.distance2clusters[c1.id]
            del c.distance2clusters[c2.id]
            c.distance2clusters[new_cluster_id] = new_cluster.distance2clusters[c.id]
        self._clusters[new_cluster_id] = new_cluster
        return new_cluster

    def fit(self, samples):
        # initialize new clusters with each new sample
        self.add_samples(samples)

        # iterate until converge
        iter_count = 0
        while True:
            iter_start_time = time.time()

            cluster_pairs = self.find_close_cluster_pairs()
            if cluster_pairs:
                for cluster_pair in cluster_pairs:
                    c1 = self._clusters[cluster_pair[0]]
                    c2 = self._clusters[cluster_pair[1]]
                    new_cluster = self.merge_cluster_pair(c1, c2)
                iter_count += 1
                logging.info('iter count {}, iter time {:.4f}s, computation count {}, mapping count {}, num pairs {}'.format(iter_count, time.time()-iter_start_time, self.computation_count, self.mapping_count, len(cluster_pairs)))
                # print('loop count {}'.format(self.loop_count))
                self.computation_count = 0
                self.mapping_count = 0
                self.loop_count = 0

            else:
                logging.info('iteration converged')
                break
        return self.changed_samples


if __name__ == '__main__':
    stopwords = []
    with open('stopwords.txt') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            stopwords.append(line.strip())
    stopwords.extend(['年', '月', '日', ' '])
    print(stopwords)

    base_dir = 'data'
    dir_names = os.listdir(base_dir)
    documents = []
    for dir_name in dir_names[:1]:
        samples = []
        file_names = os.listdir(os.path.join(base_dir, dir_name))
        print('\n' + '#'*50 + '\ndirectory name: {}, contains {} texts'.format(dir_name, len(file_names)))
        for i, file_name in enumerate(file_names):
            title, content = get_title_content(os.path.join(base_dir, dir_name, file_name))
            samples.append({'id': i, 'title': title})

        samples = preprocess(samples, True, stopwords)

        # hc = HierarchicalClustering(jaccard_distance, 0.82, False)
        hc = HierarchicalClustering(lcs_distance, 0.6, True, close_num=10)

        start_time = time.time()
        changed_samples = hc.fit(samples)
        print('training finished, time consumed {:.4f}s'.format(time.time()-start_time))

        print('changed samples', changed_samples)
        changed_clusters = defaultdict(list)
        for sample_id, cluster_id in changed_samples.items():
            changed_clusters[cluster_id].append(sample_id)
        unuseful_cluster_ids = []
        for cluster_id, sample_ids in changed_clusters.items():
            if len(sample_ids) < 2:
                unuseful_cluster_ids.append(cluster_id)
        for cluster_id in unuseful_cluster_ids:
            changed_clusters[-1].extend(changed_clusters.pop(cluster_id))
        print('changed clusters', changed_clusters)

        for cluster_id, sample_ids in hc.clusters.items():
            print('cluster {}'.format(cluster_id))
            for sample_id in sample_ids:
                print(sample_id, hc.samples[sample_id].title)
        print()

        hc.save_to('hc_model.pkl')
