import glob
import os
import sys
from itertools import combinations

import numpy as np

sys.path.append('/home/terms/bin/FACE01_IOT_dev')
from face01lib.Calc import Cal  # type: ignore
from face01lib.utils import Utils  # type: ignore

Utils_obj = Utils()
Cal_obj = Cal()

# 画像の読み込みと類似度の計算
image_dir = "predict_test"
# 画像ファイルのパスを取得
image_paths = glob.glob(os.path.join(image_dir, "*.png"))
embeddings = []

for image_path in image_paths:
    embedding = Utils_obj.get_face_encoding(image_path)
    embeddings.append(embedding)

# 類似度の計算
pairs = list(combinations(range(len(embeddings)), 2))
for i, j in pairs:
    distance = np.linalg.norm(embeddings[i] - embeddings[j])
    percent = round(Cal_obj.to_percentage(distance), 2)
    print(f'{image_paths[i]}, {image_paths[j]}, {percent}%')
    # print('---')