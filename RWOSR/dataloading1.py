import os
import json


def convert_dataset_to_json(dataset_path, output_file):
    dataset = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(category_path, filename)
                    data = {
                        'image_path': image_path,
                        'category': category
                    }
                    dataset.append(data)

    with open(output_file, 'w') as f:
        json.dump(dataset, f)


# 用法示例
dataset_path = 'data/unknow1'
output_file = 'data/unknow1/unknow1.json'
convert_dataset_to_json(dataset_path, output_file)