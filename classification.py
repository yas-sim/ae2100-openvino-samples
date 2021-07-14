import cv2
import numpy as np
from openvino.inference_engine import IECore

label = [ s.replace('\n', '') for s in open('synset_words.txt').readlines() ]
print(len(label), 'labels read')   # 読み込んだラベルの個数を表示
print(label[:20])                  # ラベルの先頭の20個を表示

# Inference Engineコアオブジェクトの生成
ie = IECore()

# IRモデルファイルの読み込み
model = './public/googlenet-v1/FP16/googlenet-v1'
net = ie.read_network(model=model+'.xml', weights=model+'.bin')

# 入出力blobの名前の取得、入力blobのシェイプの取得
input_blob_name  = list(net.inputs.keys())[0]
output_blob_name = list(net.outputs.keys())[0]
batch,channel,height,width = net.inputs[input_blob_name].shape

# モデルの情報の表示
print(input_blob_name, batch, channel, height, width)
print(output_blob_name, net.outputs[output_blob_name].shape)

# モデルをプラグインにセット
exec_net = ie.load_network(network=net, device_name='CPU', num_requests=1)

# 入力画像の前処理
img = cv2.imread('car.png')
img = cv2.resize(img, (width,height))           # スケーリング
img = img.transpose((2, 0, 1))                  # 軸の入れ替え   HWC -> CHW
img = img.reshape((1, channel, height, width))  # バッチ軸の追加 CHW -> NCHW
print('image shape=', img.shape)

# 推論実行
res = exec_net.infer(inputs={input_blob_name: img})
print('inference result shape=', res[output_blob_name].shape)

# 推論結果の表示
result=res[output_blob_name][0]
idx = np.argsort(result)[::-1]
for i in range(5):
    print(idx[i], result[idx[i]], label[idx[i]-1])
