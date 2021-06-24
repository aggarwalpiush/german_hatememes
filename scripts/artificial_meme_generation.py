#! usr/bin/env python
# -*- coding : utf-8 -*-


import pandas as pd
import codecs
import json


def main():
    deutsch_text = []
    with codecs.open('../data/deepl_translated.txt', 'r', 'utf-8') as tr_obj:
        for line in tr_obj:
            deutsch_text.append(line.replace('\n', ''))
    with codecs.open('../data/dev_seen.jsonl', 'r', 'utf-8') as ds_obj:
        new_jsons = []
        for i, line in enumerate(ds_obj):
            dev_dict = json.loads(line)
            new_json_line = dev_dict
            new_json_line['text'] = deutsch_text[i]
            new_jsons.append(new_json_line)
        with codecs.open('../data/dev_seen_deutsch.jsonl', 'w', 'utf-8') as jsonw_obj:
            for entry in new_jsons:
                json.dump(entry, jsonw_obj, ensure_ascii=False)
                jsonw_obj.write('\n')


if __name__ == '__main__':
    main()
