import requests

url="http://localhost:8000"
cfg = '50k.cfg'
weights = '50k.weights'
names = '50k.names'
img = 'dog.jpg'
# 打开文件，需要确保文件路径是正确的
with open(cfg, 'rb') as cfg_file, open(names, 'rb') as names_file, open(weights, 'rb') as weights_file, open(img, 'rb') as img_file:
    files = {
        'cfg_file': cfg_file,
        'namesfile': names_file,
        'weight_file': weights_file,
        'image_path': img_file,
    }

    # 发送 POST 请求
    response = requests.post(url, files=files)
    # response 中有文件
    with open('output.onnx', 'wb') as f:
        f.write(response.content)
    print('done')