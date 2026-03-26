import json

# JSON 파일 열기
with open("../../data/metadata/lvis_v1_train_cat_norare_info.json", "r") as f:
    data = json.load(f)

lvis_base_class = set(item["name"] for item in data if item.get("frequency") != "r")

# GPT-driven Background Concepts Among Pseudo-Background Concepts
exclude_keywords = [
    "sky", "mountain", "grass",
    "floor", "wall", "road",
    "sidewalk", "building", "water",
    "tree", "field", "sand"
]

with open('result/raw_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for img_id, img_data in data.items():
    fg = img_data.get('foreground', {})
    fg_bboxes = fg.get('bbox', [])
    fg_categories = fg.get('category', [])
    fg_captions = [item['description'] for item in fg.get('meta', [])]


    new_fg_bboxes = []
    new_fg_categories = []
    new_fg_captions = []
    moved_bboxes_to_bg = []

    for bbox, category, caption in zip(fg_bboxes, fg_categories, fg_captions):
        category_match = category.replace(' ', '_')
        if category_match in lvis_base_class or any(keyword in category for keyword in exclude_keywords):
            moved_bboxes_to_bg.append(bbox)
        else:
            new_fg_bboxes.append(bbox)
            new_fg_categories.append(category.replace(' ', '_'))
            new_fg_captions.append(caption)

    # foreground
    img_data['foreground']['bbox'] = new_fg_bboxes
    img_data['foreground']['category'] = new_fg_categories
    img_data['foreground']['caption'] = new_fg_captions

    # background
    if 'background' not in img_data:
        img_data['background'] = {'bbox': []}
    if 'bbox' not in img_data['background']:
        img_data['background']['bbox'] = []

    img_data['background']['bbox'].extend(moved_bboxes_to_bg)

with open('result/unseen_psuedo_labels.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
