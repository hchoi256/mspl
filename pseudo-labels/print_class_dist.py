import json
from collections import Counter
import matplotlib.pyplot as plt

# 파일 경로
coco_file = '../data/lvis_v1/annotations/lvis_v1_train_base.json'
output_img_path = 'tmp.png'

# COCO JSON 파일 불러오기
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# category_id ➝ name 매핑
category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
name_to_category_id = {v: k for k, v in category_id_to_name.items()}

# person 카테고리 ID 찾기
person_id = name_to_category_id.get('person')

# 카테고리 ID별 annotation 수 세기 (person 제외)
category_counts = Counter()
for ann in coco_data['annotations']:
    if ann['category_id'] != person_id:
        category_counts[ann['category_id']] += 1

# 개수 기준 정렬
sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

# 이름과 개수 리스트 생성
category_names = [category_id_to_name[cat_id] for cat_id, _ in sorted_categories]
counts = [count for _, count in sorted_categories]


# 4. 개수 기준으로 오름차순 정렬해서 하위 10개 추출
lowest_10 = sorted(category_counts.items(), key=lambda x: x[1])[:10]

print("📉 주석 개수 기준 하위 10개 클래스:")
for cat_id, count in lowest_10:
    name = category_id_to_name.get(cat_id, f"Unknown (id: {cat_id})")
    print(f"{name} (id: {cat_id}): {count} annotations")

# 리스트 형태로도 출력
lowest_10_names = [category_id_to_name[cat_id] for cat_id, _ in lowest_10]
print("\n하위 10개 클래스 리스트:")
print(lowest_10_names)

# 시각화
plt.figure(figsize=(14, 6))
plt.bar(category_names, counts)
plt.xticks(rotation=90)
plt.xlabel('Category (excluding person)')
plt.ylabel('Number of Annotations')
plt.title('COCO Category Distribution (No "person")')
plt.tight_layout()

# 저장
plt.savefig(output_img_path, dpi=300)
plt.close()

print(f"✅ 'person' 제외한 클래스 분포 시각화를 '{output_img_path}' 파일로 저장했습니다.")
