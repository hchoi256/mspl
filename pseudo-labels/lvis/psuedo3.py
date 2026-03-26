import json
from collections import defaultdict

with open('result/unseen_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bbox_count = defaultdict(int)
total_annotations = 0

for img_id, img_data in data.items():
    categories = img_data.get('foreground', {}).get('category', [])
    for category in categories:
        bbox_count[category] += 1
        total_annotations += 1

threshold = 1
filtered_classes = [(cat, cnt) for cat, cnt in bbox_count.items() if cnt >= threshold]
filtered_classes = sorted(filtered_classes, key=lambda x: x[1], reverse=True)

top_class_list = [category for category, _ in filtered_classes]
print(top_class_list, total_annotations)