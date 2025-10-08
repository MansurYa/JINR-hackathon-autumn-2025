import json

import requests

def read_json(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f_in:
        data = json.load(f_in)

    return data


def insert_one(elastic_host: str, index_name: str, item: dict):
    post_url = elastic_host + index_name + '/' + '_doc'
    r = requests.post(post_url, json=item, verify=False)
    if r.status_code != 201:
        print(r.text, r.status_code)
        raise ValueError('Data mapping mismatch')


def update_mapping(elastic_host, index_name, mapping=None):
    mapping_url = elastic_host + '/'.join([index_name, '_mapping'])
    print(mapping_url)
    r = requests.put(mapping_url, json=mapping, verify=False)
    print(r.text, r.status_code)


def update_settings(elastic_host, index_name, settings=False):
    if settings:
        settings_url = elastic_host + '/'.join([index_name, '_settings'])
        r = requests.put(settings_url, json=settings, verify=False)
        print(r.text, r.status_code)


def create_index(elastic_host, index_name):
    index_url = elastic_host + index_name
    r = requests.put(index_url, verify=False)
    print(r.text, r.status_code)

def insert_batch(elastic_host, index_name, data):
    upload_url = elastic_host + '/' + '_bulk'
    chunk = 100
    action = {'create': {'_index': index_name}}
    header = {'Content-Type': 'application/x-ndjson'}
    parts = len(data) // chunk
    if len(data) % chunk != 0:
        parts += 1
    for i in range(parts):
        post_data = ''
        values = data[max(0, chunk*i):min(chunk*(i+1), len(data))]
        for value in values:
            post_data += json.dumps(action) + '\n'
            post_data += json.dumps(value) + '\n'
        r = requests.post(upload_url, headers=header, verify=False, data=post_data)
        print(r.status_code)

def main():
    login = 'elastic'
    password = 'elastictest'
    index = 'data_27'
    host = f'http://{login}:{password}@144.124.229.26:5601/'

    create_index(host, index)
    index_settings = read_json("../temporary_data/mapping.json")
    update_settings(host, index, index_settings['settings'])
    update_mapping(host, index, index_settings['mappings'])
    data = read_json('../temporary_data/data.json')
    insert_batch(host, index, data)
    # for item in data:
    #     insert_one(host, index, item=item)


if __name__ == '__main__':
    main()
