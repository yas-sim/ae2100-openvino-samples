import cv2
import numpy as np
from openvino.inference_engine import IECore

label = [ s.replace('\n', '') for s in open('voc_labels.txt').readlines() ]
print(len(label), 'labels read')   # 読み込んだラベルの個数を表示
print(label)                       # ラベルを表示

# Inference Engineコアオブジェクトの生成
ie = IECore()

# IRモデルファイルの読み込み
model = './public/mobilenet-ssd/FP16/mobilenet-ssd'
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
img    = cv2.imread('resources/car_1.bmp')
in_img = cv2.resize(img, (width,height))
in_img = in_img.transpose((2, 0, 1))
in_img = in_img.reshape((1, channel, height, width))
print('image shape=', in_img.shape)

# 推論実行
res = exec_net.infer(inputs={input_blob_name: in_img})
print('inference result shape=', res[output_blob_name].shape)

# 推論結果の処理 (バウンディングボックス、ラベル描画)
result = res[output_blob_name][0][0]
img_h, img_w, _ = img.shape
for obj in result:
    imgid, clsid, confidence, x1, y1, x2, y2 = obj
    if confidence>0.6:          # 確かさが60%以上のオブジェクトを描画
        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x2 = int(x2 * img_w)
        y2 = int(y2 * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), thickness=4 )
        cv2.putText(img, label[int(clsid)], (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,255,255), thickness=4)

# 結果の表示 (GUI環境が必要)
cv2.imshow('image', img)
cv2.waitKey(3*1000)
