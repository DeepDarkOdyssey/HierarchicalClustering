# -*- coding: utf-8 -*-

from collections import defaultdict
import time
import pickle
import logging
from munch import Munch


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')


class HierarchicalClustering(object):
    """层次聚类模型类
    提供类似sklearn接口的类，实现了层次聚类的算法，使用self.fit(sample)增量的加载输入的样本进行训练

    Args:
        name: 一个str, 模型的名称
        dist_func: 一个callable, 计算两个字符串之间距离度量的函数，接受两个字符串s1, s2作为输入
        thresh_hold: 一个float, 代表距离度量的阈值，小于该阈值的两个cluster才会被聚合
        use_buckets: bool，是否需要使用分桶
        close_num: int, 每次迭代聚类算法最大允许合并的cluster对的数量
    """
    def __init__(self, name, dist_func, threshold, use_buckets=False, close_num=10):
        self.name = name
        self.dist_func = dist_func
        self.threshold = threshold
        self.use_buckets = use_buckets
        self.close_num = close_num

        # 一个dict，将实际的cluster的id映射到其对应的sample的id构成的list上，例如：
        # {cluster_id1: [sample_id1, sample_id2 ... ], cluster_id2: [...], ...}
        self._clusters = {}
        self.samples = {}   # 一个dict，将sample的id映射到其对应的sample上，例如：{sample_id1: sample1, sample_id2: sample2}
        self.sample_ids = []    # 一个list，记录该模型实例中已加载过的所有的sample的id
        self._current_cluster_id = -1   # 初始的cluster id
        self.bucket2id = defaultdict(list)  # 一个dict, 记录每个token对应哪些sample的id
        self.id2bucket = defaultdict(list)  # 一个dict, 记录每个sample的id包含哪些token
        self._changed_sample_ids = set()    # 记录在一次调用self.fit()函数中所属cluster发生变化的所有sample的id

    @property
    def clusters(self):
        """将模型内部实际的簇self._clusters转换为展示用的cluster，将所有的只包含一个sample的簇并为一个id为-1的簇并返回

        Returns:
            一个代表cluster的dict，将cluster的id映射到其对应的sample的id构成的list上，例如：
            {cluster_id1: [sample_id1, sample_id2 ... ], cluster_id2: [...], ...}
        """
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
        """将模型记录的当前最大cluster id自增1，并返回

        Returns:
            一个表示新加入的cluster的id的int
        """
        self._current_cluster_id += 1
        return self._current_cluster_id

    @property
    def changed_samples(self):
        """返回模型记录的有变动的sample的id及其对应的cluster的id，并清空变动sample的id的记录

        Returns:
            一个dict, 将sample的id映射到其对应的cluster的id，例如：
            {sample_id1: cluster_id1, sample_id2: cluster_id2, ...}
        """
        changed_samples = {}
        for sample_id in self._changed_sample_ids:
            changed_samples[sample_id] = self.samples[sample_id].cluster_id
        self._changed_sample_ids.clear()
        return changed_samples

    def clear(self):
        """清空所有的实例变量"""

        self._clusters = {}
        self.samples = {}
        self.sample_ids = []
        self._current_cluster_id = -1
        self.bucket2id = defaultdict(list)
        self.id2bucket = defaultdict(list)
        self._changed_sample_ids = set()

    def save_to(self, file_path):
        """将模型实例保存到file_path的位置

        Args:
            file_path: string, 想要保存模型实例的文件地址
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_from(self, file_path):
        """从file_path加载已保存的模型实例并覆盖当前实例的所有属性

        Args:
            file_path: string, 想要加载的模型实例的文件地址
        """
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
        """将模型中所有与输入的sample_ids相关的属性、信息完全删除

        Args:
            sample_ids: list of int, 想要删除的sample的id构成的list
        """
        for sample_id in sample_ids:
            correlated_sample = self.samples[sample_id]     # 找出sample_id对应的sample
            self.sample_ids.remove(sample_id)       # 从self.sample_ids中删除sample_id
            self.samples.pop(sample_id)         # 从self.samples中删除对应的sample
            for sample in self.samples:
                sample.distance2samples.pop(sample_id)      # 将self.samples中与sample相关的所有距离信息删除

            correlated_cluster = self._clusters[correlated_sample.cluster_id]   # 找出sample属于的cluster
            correlated_cluster.sample_ids.remove(sample_id)                     # 将sample_id从cluster中删除
            for cluster_id, cluster in self._clusters.items():
                if cluster_id != correlated_cluster.id:
                    self.compute_cluster_distance(correlated_cluster, cluster)  # 更新self._clusters中与cluster相关的距离信息

    def add_samples(self, samples):
        """将输入的samples添加到模型实例的属性中

        Args:
            samples: 由表征sample的dict构成的list，每个sample的结构为：
                    {'id': (int), 'title': title(sting), 'seg_title': segments(list of string),
                    'cluster_id': cluster_id(int), distance2samples: {} }

        Raises:
            IndexError: sample已经在模型中
        """
        for sample in samples:
            if sample.id in self.sample_ids:    # 查询sample的id是否已经在模型中
                raise IndexError('The sample {} has already been added to the model'.format(sample.id))
            else:
                self.sample_ids.append(sample.id)
                self._changed_sample_ids.add(sample.id)
                self.samples[sample.id] = sample

                if sample.cluster_id is None:
                    # 如果sample没有参与过聚类，并不属于任何一个cluster，获取一个新的cluster的id
                    cluster_id = self.current_cluster_id
                    sample.cluster_id = cluster_id
                if sample.cluster_id not in self._clusters:
                    # 如果sample属于的cluster的id并不在模型实例的记录中，用该id创建一个新的cluster加入到模型中, cluster的结构如下
                    cluster_dict = {'id': sample.cluster_id, 'sample_ids': [sample.id], 'distance2clusters': {}}
                    self._clusters[sample.cluster_id] = Munch(cluster_dict)
                if self.use_buckets and sample.seg_title:
                    # 给每个sample按其分词得到的token进行分桶
                    for token in sample.seg_title:
                        self.bucket2id[token].append(sample.id)
                        self.id2bucket[sample.id].append(token)

    def compute_sample_distance(self, s1, s2):
        """计算两个sample之间的距离"""

        # 查询两个sample之间是否互相计算过距离，是的话省略计算步骤
        if s2.id in s1.distance2samples:
            dist = s1.distance2samples[s2.id]

        elif s1.id in s2.distance2samples:
            dist = s2.distance2samples[s2.id]

        else:
            # 需要进行距离计算的话，先判断两个sample是否属于同一个桶
            if self.use_buckets and len(self.bucket2id) > 0:
                in_same_bucket = False
                dist = 1
                for bucket in self.id2bucket[s1.id]:
                    if s2.id in self.bucket2id[bucket]:
                        in_same_bucket = True
                        break
                if in_same_bucket:
                    # 将string转为list of int, 是为了让numba.jit的加速效果生效
                    dist = self.dist_func([ord(c) for c in s1.title], [ord(c) for c in s2.title])
            else:
                dist = self.dist_func([ord(c) for c in s1.title], [ord(c) for c in s2.title])

            # 将计算过的距离进行保存
            s1.distance2samples[s2.id] = dist
            s2.distance2samples[s1.id] = dist
        return dist

    def compute_cluster_distance(self, c1, c2):
        """计算两个cluster之间的距离

        Args:
            c1: 一个cluster
            c2: 另一个cluster, cluster的结构为：
                {'id': cluster_id(int), 'sample_ids': [id1, id2, ...](list of int), 'distance2clusters': {}}


        Returns:
            一个float代表两个cluster之间的距离，0最小1最大
        """
        # 查询两个cluster之间是否计算过距离
        if c2.id in c1.distance2clusters:
            dist = c1.distance2clusters[c2.id]

        elif c1.id in c2.distance2clusters:
            dist = c2.distance2clusters[c1.id]

        else:
            # 计算两个cluster之间的距离，即两个cluster包含的sample两两之间的最大值
            distances = []
            for s1_id in c1.sample_ids:
                for s2_id in c2.sample_ids:
                    distance = self.compute_sample_distance(self.samples[s1_id], self.samples[s2_id])
                    distances.append(distance)
            dist = max(distances)

            # 将得到距离进行保存，避免重复查询
            c1.distance2clusters[c2.id] = dist
            c2.distance2clusters[c1.id] = dist
        return dist

    # def compute_cluster_distance_prev(self, c1, c2):
    #     distances = []
    #     for s1_id in c1.sample_ids:
    #         for s2_id in c2.sample_ids:
    #             distance = self.compute_sample_distance(self.samples[s1_id], self.samples[s2_id])
    #             distances.append(distance)
    #     dist = max(distances)
    #     c1.distance2clusters[c2.id] = dist
    #     c2.distance2clusters[c1.id] = dist
    #     return dist

    def find_nearest_cluster_pair(self):
        """寻找模型已保存的cluster中最相近的一对

        Returns:
            一个tuple，包含两个最相近的cluster的id, 例如（cluster_id1, cluster_id2）
        """
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
                    # self.loop_count += 1
                    dist = self.compute_cluster_distance(cluster, another_cluster)
                    if dist < min_dist and dist <= self.threshold:
                        cluster_pair = (cluster_id, another_id)
                        min_dist = dist
        return cluster_pair

    def find_close_cluster_pairs(self):
        """寻找模型已保存的cluster中最相近的n对

        Returns:
            一个list of tuple，每个tuple包含两个最相近的cluster的id

        """
        close_cluster_pairs = []
        clusters = self._clusters.copy()
        while True:
            try:
                cluster_id, cluster = clusters.popitem()
            except KeyError:
                break
            for another_id, another_cluster in clusters.items():
                dist = self.compute_cluster_distance(cluster, another_cluster)
                if dist <= self.threshold:      # 只保存距离小于阈值的cluster pair
                    close_cluster_pairs.append((dist, (cluster_id, another_id)))

        cluster_pairs = []

        added_cluster_ids = []
        # 按距离从小到大对close_cluster_pairs排序，取不重复的前self.close_num个返回
        for dist, cluster_pair in sorted(close_cluster_pairs):
            if cluster_pair[0] not in added_cluster_ids and cluster_pair[1] not in added_cluster_ids:
                cluster_pairs.append(cluster_pair)
                added_cluster_ids.extend(cluster_pair)
        return cluster_pairs[:self.close_num]

    def merge_cluster_pair(self, c1, c2):
        """将输入的两个cluster合并后返回"""

        new_cluster_id = self.current_cluster_id    # 先获取要创建的新cluster的id
        new_cluster = Munch(id=new_cluster_id, sample_ids=[], distance2clusters={})
        # 将要合并的两个cluster对应的sample_id, distance2clusters从模型中删除并加入到新的cluster中
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
        """将输入的sample加入模型并迭代到收敛，返回本次训练中cluster发生变化的sample

        Args:
            samples: list of samples, sample的结构见self.add_samples()的doc string

        Returns:
            一个set, 记录在本次训练中cluster发生变化的所有sample的id
        """
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
                logging.info('iter {}, iter time {:.4f}s, num pairs {}'.format(
                    iter_count, time.time()-iter_start_time, len(cluster_pairs))
                )

            else:
                logging.info('iteration converged')
                break
        return self.changed_samples
