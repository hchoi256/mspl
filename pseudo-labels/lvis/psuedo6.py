import json
from collections import defaultdict

cat_info_file = '../../data/metadata/lvis_v1_train_cat_norare_info.json'
pseudo_label_file = 'result/topk_unseen_psuedo_labels.json'
output_file = 'result/lvis_v1_train_cat_norare_info_pseudo_v0_new_caption.json'

with open(cat_info_file, 'r', encoding='utf-8') as f:
    class_info = json.load(f)
name_to_info = {cls['name']: cls for cls in class_info}
max_cat_id = max(cls['id'] for cls in class_info)
lvis_base_class_ids = [cls['id'] for cls in class_info]

with open(pseudo_label_file, 'r', encoding='utf-8') as f:
    pseudo_data = json.load(f)

instance_counter = defaultdict(int)
image_counter = defaultdict(set)

for img_id, info in pseudo_data.items():
    categories = info.get('foreground', {}).get('category', [])
    for cat in categories:
        instance_counter[cat] += 1
        image_counter[cat].add(img_id)

for cat_name in instance_counter:
    inst_count = instance_counter[cat_name]
    img_count = len(image_counter[cat_name])

    if cat_name in name_to_info:
        name_to_info[cat_name]['instance_count'] += inst_count
        name_to_info[cat_name]['image_count'] += img_count
    else:
        max_cat_id += 1
        name_to_info[cat_name] = {
            'name': cat_name,
            'instance_count': inst_count,
            'image_count': img_count,
            'id': max_cat_id,
            'frequency': 'c',  # default
            'synonyms': [cat_name],
            'def': 'pseudo label class',
            'synset': f'{cat_name.replace(" ", "_")}.n.01'
        }

updated_class_info = list(name_to_info.values())
updated_class_info = sorted(updated_class_info, key=lambda x: x['id'])
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(updated_class_info, f, ensure_ascii=False, indent=2)
