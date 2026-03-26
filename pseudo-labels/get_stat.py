import json
from collections import defaultdict

# ========== [1] 의사라벨 JSON 로딩 및 클래스별 bbox 수 세기 ==========
with open('unseen_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bbox_count = defaultdict(int)
total_annotations = 0

for img_id, img_data in data.items():
    categories = img_data.get('foreground', {}).get('category', [])
    for category in categories:
        bbox_count[category] += 1
        total_annotations += 1

# ========== [2] threshold 이상 클래스 필터링 ==========
threshold = 1
filtered_classes = [(cat, cnt) for cat, cnt in bbox_count.items() if cnt >= threshold]
filtered_classes = sorted(filtered_classes, key=lambda x: x[1], reverse=True)

# 출력
print(f"=== bbox 수가 {threshold}개 이상인 클래스 ===")
for category, count in filtered_classes:
    print(f"{category}: {count} bboxes")

# 리스트 형태로 저장
top_class_list = [category for category, _ in filtered_classes]
print("\n[bbox 수 기준 top 클래스 리스트]")
print(top_class_list)
print(f"총 top 클래스 수: {len(top_class_list)}")

print(f"\n총 annotation 개수: {total_annotations}")

# ========== [3] 전체 클래스와 비교 ==========
# 전체 클래스 정보 로딩 (예: 1203개 클래스)
with open('../../../data/metadata/lvis_v1_train_cat_norare_info.json', 'r', encoding='utf-8') as f:
    category_meta = json.load(f)

all_class_names = set(cat['name'] for cat in category_meta)
print(f"\n전체 클래스 수: {len(all_class_names)}")  # 보통 1203

# 의사라벨에 등장한 클래스
pseudo_label_classes = set(top_class_list)

# 등장한 클래스 개수 계산
used_classes = all_class_names.intersection(pseudo_label_classes)
print(f"\n의사라벨에서 등장한 클래스 수: {len(used_classes)}")
print("등장한 클래스 예시:", list(used_classes)[:10])  # 앞 10개 예시

# 등장하지 않은 클래스
unused_classes = all_class_names - pseudo_label_classes
print(f"의사라벨에서 등장하지 않은 클래스 수: {len(unused_classes)}")
