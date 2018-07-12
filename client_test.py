import requests
import json
import pandas as pd
import os
import random
from utils import get_title_content


# samples = []
# base_dir = 'data'
# dir_names = os.listdir(base_dir)
# for dir_name in dir_names:
#     file_names = os.listdir(os.path.join(base_dir, dir_name))
#     for i, file_name in enumerate(file_names):
#         title, content = get_title_content(os.path.join(base_dir, dir_name, file_name))
#         samples.append({'id': i, 'title': title, 'model_name': dir_name})
#
# random.shuffle(samples)

url = 'http://127.0.0.1:10000/'
r = requests.get(url)
print(r.text)

df = pd.DataFrame.from_csv('real_news.csv')
for model_name, sub_df in df.groupby('model_name'):
    samples = sub_df.to_dict('records')
    json_string = json.dumps(samples, ensure_ascii=False)
    r = requests.post(url + 'update/{}'.format(model_name), json=json_string)
    print(r.text)

r = requests.get(url + 'showall')
print(r.text)


