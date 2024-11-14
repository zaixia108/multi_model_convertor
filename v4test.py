import cv2
import onnxruntime

model = 'output.onnx'
ort = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider'])

img = cv2.imread('img_5.png')

width = img.shape[1]
height = img.shape[0]
oimg = img.copy()

img = cv2.resize(img, (416, 416))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))
img = img.reshape(1, 3, 416, 416)
img = img.astype('float32')

# outname 1"box 2"conf

output_names = ort.get_outputs()
names = []
for i in output_names:
    names.append(i.name)
# print(names)

outputs = ort.run(None, {'input': img})
# print(outputs)
boxex = outputs[0][0]
conf = outputs[1][0]

print(len(boxex))
print(len(conf))

for i in range(len(boxex)):

    conf_all = sum(conf[i])
    if conf_all > 0.5:
        print(boxex[i], conf[i])
        box = boxex[i][0]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        cv2.rectangle(oimg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.putText(img, str(conf_all), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        pass

cv2.imshow('img', oimg)
cv2.waitKey(0)