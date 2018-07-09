import requests
import json
import os
import random
from utils import get_title_content


samples = []
base_dir = 'data'
dir_names = os.listdir(base_dir)
for dir_name in dir_names:
    file_names = os.listdir(os.path.join(base_dir, dir_name))
    for i, file_name in enumerate(file_names):
        title, content = get_title_content(os.path.join(base_dir, dir_name, file_name))
        samples.append({'id': i, 'title': title, 'model_name': dir_name})

random.shuffle(samples)
print(samples)

url = 'http://127.0.0.1:10000/'

r = requests.get(url)
print(r.text)

for sample in samples:
    model_name = sample.pop('model_name')

    json_string = json.dumps([sample], ensure_ascii=False)
    r = requests.post(url + 'update/{}'.format(model_name), json=json_string)
    print(r.text)

r = requests.get(url + 'showall')
print(r.text)


