import os
from itertools import combinations

import numpy as np
import onnx
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image

# model_name = 'model_100epoch_128dim.onnx'
# model_name = 'model_169epoch_512dim.onnx'
# model_name = 'signed_model.onnx'
model_name = 'efficientnetv2_arcface.onnx'
optimal_threshold = 0.4


# 画像の前処理を定義
mean_value = [0.485, 0.456, 0.406]
std_value = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean_value,
        std=std_value
    )
])

# ONNXモデルをロード
onnx_model = onnx.load(model_name)
ort_session = ort.InferenceSession(model_name)

# 署名表示
for prop in onnx_model.metadata_props:
    if prop.key == "signature":
        print(prop.value)

# 入力名を取得
input_name = onnx_model.graph.input[0].name

# 推論対象の画像ファイルを取得
image_dir = "predict_test"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

# 類似度判断の関数
def is_same_person(embedding1, embedding2, threshold):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cos_sim >= threshold, cos_sim

# 百分率の計算
def percentage(cos_sim):
    return round(-23.71 * cos_sim ** 2 + 49.98 * cos_sim + 73.69, 2)

# 画像を読み込み、前処理を行い、モデルで推論を行う
embeddings = []
for image_file in image_files:
    image = Image.open(image_file).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # バッチ次元を追加
    image = image.numpy()
    embedding = ort_session.run(None, {input_name: image})[0]  # 'input'をinput_nameに変更
    embeddings.append(embedding)

# # 類似度の計算
# pairs = list(combinations(range(len(embeddings)), 2))
# for i, j in pairs:
#     similarity, cos_sim = is_same_person(embeddings[i], embeddings[j], optimal_threshold)
#     if similarity == True:
#         print(f"Similarity between {image_files[i]} and {image_files[j]}: {similarity}, {cos_sim}")


# すべての画像ペアの類似度を計算
for (file1, embedding1), (file2, embedding2) in combinations(zip(image_files, embeddings), 2):
    similarity, cos_sim = is_same_person(embedding1, embedding2, optimal_threshold)
    print(f"{file1}, {file2}, {similarity}, {percentage(cos_sim)}%")
