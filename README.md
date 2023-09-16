<div align="center">

<img src="https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/images/g1320.png" width="200px">

日本人用顔認証学習モデル
___
</div>

[English](README_english.md)

# ℹ️ Note
- ここに置いてある学習モデルについて。
  - 日本人の顔認証を目的としています。
  - 無料でお使いいただけます。ご使用時に著作権表示をして下さい。
  - 商用利用されたい場合はライセンスが必要です。

# 日本人顔認識のための新たな学習モデル `JAPANESE FACE v1`

- [ℹ️ Note](#ℹ️-note)
- [日本人顔認識のための新たな学習モデル `JAPANESE FACE v1`](#日本人顔認識のための新たな学習モデル-japanese-face-v1)
  - [はじめに](#はじめに)
  - [Dlibの歴史と特徴](#dlibの歴史と特徴)
  - [`dlib_face_recognition_resnet_model_v1.dat`](#dlib_face_recognition_resnet_model_v1dat)
  - [作成方法](#作成方法)
  - [`dlib_face_recognition_resnet_model_v1.dat`と`JAPANESE FACE v1`の性能評価](#dlib_face_recognition_resnet_model_v1datとjapanese-face-v1の性能評価)
    - [一般日本人に対しての性能評価](#一般日本人に対しての性能評価)
    - [若年日本人女性に対しての性能評価](#若年日本人女性に対しての性能評価)
- [`JAPANESE FACE v1`について](#japanese-face-v1について)
  - [一般日本人に対しての性能評価](#一般日本人に対しての性能評価-1)
  - [若年日本人女性に対しての性能評価](#若年日本人女性に対しての性能評価-1)
  - [使い方](#使い方)
  - [`dlib`学習モデルとの比較](#dlib学習モデルとの比較)
    - [`dlib_predict.py`](#dlib_predictpy)
    - [実行結果](#実行結果)
  - [まとめ](#まとめ)
  - [See also](#see-also)


## はじめに

顔認識技術は、スマートフォンのロック解除から空港のシステムまで、私たちの生活のあらゆる面で使用されています。しかしOSSの既存顔認識モデルである[Dlib](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)の`dlib_face_recognition_resnet_model_v1.dat`は、白人の顔に対する精度は高いものの、それ以外の人種、とくに若年日本人女性の顔に対する精度が低いという問題があります。

## Dlibの歴史と特徴

Dlibは、元々はC++で書かれた機械学習とデータ解析のためのオープンソースライブラリです。2002年にDavis Kingによって開発が始まりました。Dlibは、顔認証の分野でよく知られていますが、その機能はそれだけにとどまりません。このライブラリは、画像処理、機械学習、自然言語処理、数値最適化といった多様なタスクに対応しています。C++で開発された本ライブラリはPythonバインディングも提供しています。[dlib(GitHub)](https://github.com/davisking/dlib)で、現在も開発が続けられています。

## `dlib_face_recognition_resnet_model_v1.dat`

このモデルは、ResNetベースの深層学習モデルで、非常に高い精度で顔認証が可能です。2017年に提供が開始されました。
Labeled Faces in the Wild (LFW) データセットでの精度は99.38%と報告されています。このような高い精度が、Dlibとその顔認証モデルが広く採用される一因です。

しかしこのモデルは白人に対する精度は高いものの、それ以外の人種、とくに若年日本人女性に対する精度は低いという問題がありました。そのため、これまでは認識精度を向上させるためにしきい値を調整するなどの対策が必要でした。またしきい値を調整してもなお`False Positive (偽陽性)`が一定値存在する問題が残っていました。

<div align="center">
<img src="img/image845.png" alt="Dlibが偽陽性を出す例"><br />
Dlibが偽陽性を出す例
</div>

そこで、日本人の顔データセットを使用して新たな顔認識モデルを開発することにしました。

結果として新たに開発した学習モデルは`26MB`と、既存のdlibモデル（`22.5MB`）よりわずかに大きいサイズとなりましたが、同等の計算リソースでより高精度な日本人の顔認証が可能となりました。

## 作成方法

大規模な画像データセットである`ImageNet`で事前学習された`EfficientNetV2`を、`ArcFace`を使い、日本人の顔データセットでファインチューニングしました。
これにより、特徴空間での分離を改善し、日本人の顔認識精度を大幅に向上できました。

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)


UMAPによる可視化結果
![UMAPによる可視化結果](img/umap.png)


## `dlib_face_recognition_resnet_model_v1.dat`と`JAPANESE FACE v1`の性能評価
これら顔学習モデルの性能を比較・評価するために、以下の検証を行いました。なお、使用する顔データセットは、`JAPANESE FACE v1`の作成に使用したデータセットには含まれていません。

### 一般日本人に対しての性能評価
著名日本人の顔画像データベースから、ランダムに300枚の画像を選択し、一般日本人の顔画像データセットを作成しました。このデータセットに対して、`dlib_face_recognition_resnet_model_v1.dat`を用いて顔認証を行い、ROC-AUCを計算しました。その結果が以下になります。
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/一般日本人.png)
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/一般日本人_dlib_ROC.png)
一般日本人に対して、`dlib_face_recognition_resnet_model_v1.dat`のAUCは0.98であり、非常に高い精度を示しています。

### 若年日本人女性に対しての性能評価
今度は、著名日本人の顔画像データベースから、**とくに若年女性の顔画像を**ランダムに300枚選択し、若年日本人女性の顔画像データセットを作成しました。このデータセットに対して、`dlib_face_recognition_resnet_model_v1.dat`を用いて顔認証を行い、ROC-AUCを計算しました。その結果が以下になります。
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/若年日本人女性.png)
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/若年日本人女性_dlib_ROC.png)
一般日本人に対して、若年日本人女性の顔画像を用いて性能評価をしたところ、AUCが0.98から0.94に低下しました。
これはDlibの顔学習モデルが、face scrub datasetやVGGデータセットを主に使用しているところが原因と考えられます。これらのデータセットには、若年日本人女性の顔画像がほとんど含まれていないため、若年日本人女性の顔画像に対しては、性能が低下すると考えられます。（[High Quality Face Recognition with Deep Metric Learning](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)を参照）

# `JAPANESE FACE v1`について
この問題を解決するため、独自の日本人顔データセットを用いて学習したモデルが`JAPANESE FACE v1`です。このモデルは`EfficientNetV2`に`ArcFaceLoss`を適用して作成されました。作成の詳細は、[日本人顔認識のための新たな学習モデルを作成 ~ `EfficientNetV2`ファインチューニング ~](https://zenn.dev/ykesamaru/articles/bc74ec27925896)という記事で詳しく解説しています。
このモデルを使って、Dlibの学習モデルと比較した結果を以下に示します。

## 一般日本人に対しての性能評価
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/一般日本人_dlib_vs_japaneseFace_ROC.png)
Dlibの学習モデルと比較して、AUCが0.98であり、同等の性能を示しています。ROC曲線をみると、`JAPANESE FACE v1`の方が`dlib`よりも、左上の部分が凸となっており、性能が高いことがわかります。

## 若年日本人女性に対しての性能評価
![](https://raw.githubusercontent.com/yKesamaru/dlib_vs_japaneseFace/master/img/若年日本人女性_dlib_vs_japaneseFace_ROC.png)
若年日本人女性の顔画像に対しては、DlibのAUCが0.94に対し、`JAPANESE FACE v1`は0.98を維持しています。

## 使い方
https://github.com/yKesamaru/FACE01_SAMPLE
上記URLから`FACE01`をインストールしてください。
`setup.py`による一括インストールや、`Docker`での使用ができます。
インストールが終わったら、以下のコードを実行してください。

```bash
# 仮想環境のアクティベート
$ source ./bin/activate
# Exampleコードの実行
$ python example/simple_efficientnetv2_arcface.py
```


## `dlib`学習モデルとの比較
[`dlib_face_recognition_resnet_model_v1.dat`を用いた場合全て同一人物と判断されてしまう例 (不正解) ](https://tokai-kaoninsho.com/%e3%82%b3%e3%83%a9%e3%83%a0/%e9%a1%94%e8%aa%8d%e8%a8%bc%e3%82%92%e4%bd%bf%e3%81%a3%e3%81%9f%e3%80%8c%e9%a1%94%e3%81%8c%e4%bc%bc%e3%81%a6%e3%82%8b%e8%8a%b8%e8%83%bd%e4%ba%ba%e3%83%a9%e3%83%b3%e3%82%ad%e3%83%b3%e3%82%b0%e3%80%8d/)を、新しく作成した学習モデルで検証しました。**新しいモデルでは全て別人と判断 (正解) されました**。

### `dlib_predict.py`
```python
import glob
import os
import sys
from itertools import combinations

import numpy as np

sys.path.append('FACE01')
from face01lib.Calc import Cal
from face01lib.utils import Utils

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
```

### 実行結果
- 新井浩文, 大森南朋
- ![](img/PASTE_IMAGE_2023-06-17-22-28-51.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/新井浩文.png_align_resize.png, predict_test/大森南朋.png_align_resize.png, False, 87.98%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/新井浩文.png_align_resize.png, predict_test/大森南朋.png_align_resize.png, 98.97%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 新川優愛, 内田理央
- ![](img/PASTE_IMAGE_2023-06-17-22-29-28.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - ` predict_test/新川優愛.png, predict_test/内田理央.png, False, 81.46%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/内田理央.png_align_resize.png, predict_test/新川優愛.png_align_resize.png, 99.27%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 金正恩, 馬場園梓
- ![](img/PASTE_IMAGE_2023-06-17-22-30-12.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/金正恩.png_align_resize.png, predict_test/馬場園梓.png_align_resize.png, False, 79.87%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/金正恩.png_align_resize.png, predict_test/馬場園梓.png_align_resize.png, 99.44%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 池田清彦, 西村康稔
- ![](img/PASTE_IMAGE_2023-06-17-22-30-39.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/池田清彦.png_align_resize.png, predict_test/西村康稔.png_align_resize.png, False, 72.26%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/池田清彦.png_align_resize.png, predict_test/西村康稔.png_align_resize.png, 98.87%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 金正恩, 畑岡奈紗
- ![](img/PASTE_IMAGE_2023-06-17-22-31-12.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  `predict_test/金正恩.png_align_resize.png, predict_test/畑岡奈紗.png_align_resize.png, False, 77.82%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/金正恩.png_align_resize.png, predict_test/畑岡奈紗.png_align_resize.png, 99.37%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 有働由美子, 椎名林檎
- ![](img/PASTE_IMAGE_2023-06-17-22-31-42.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/有働由美子.png_align_resize.png, predict_test/椎名林檎.png_align_resize.png, False, 82.17%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  -`predict_test/有働由美子.png_align_resize.png, predict_test/椎名林檎.png_align_resize.png, 99.16%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 波瑠, 入山杏奈
- ![](img/PASTE_IMAGE_2023-06-17-22-32-17.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/波瑠.png_align_resize.png, predict_test/入山杏奈.png_align_resize.png, False, 81.46%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/波瑠.png_align_resize.png, predict_test/入山杏奈.png_align_resize.png, 99.07%`
  - 判定: 同一人物 **(不正解:x:)**
---
- 浅田舞, 浅田真央
- ![](img/PASTE_IMAGE_2023-06-17-22-32-38.png)
- 新しい学習モデル (`efficientnetv2_arcface.onnx`)
  - `predict_test/浅田舞.png_align_resize.png, predict_test/浅田真央.png_align_resize.png, False, 83.06%`
  - 判定; 別人 **(正解:o:)**
- 既存の学習モデル (`dlib_face_recognition_resnet_model_v1.dat`)
  - `predict_test/浅田舞.png_align_resize.png, predict_test/浅田真央.png_align_resize.png, 99.27%`
  - 判定: 同一人物 **(不正解:x:)**
---

## まとめ

この記事では日本人の顔認識の精度を向上させるために、`EfficientNetV2`と`ArcFace`を用いた新たなモデルの開発について説明しました。

この新たなモデルは日本人の顔認識において、既存のモデル( `dlib_face_recognition_resnet_model_v1.dat`) よりも優れた性能を示しました。これにより顔認識技術がさらに多様なシチュエーションに対応できるようになり、その応用範囲が広がることが期待されます。


## See also
- [FACE01](https://github.com/yKesamaru/FACE01_SAMPLE)
  - Face recognition library that integrates various functions and can be called from Python.
