import paddlex as pdx
import cv2

print("Loading model...")
model = pdx.load_model('path_to_model')
print("Model loaded.")

im = cv2.imread('test.jpg') # 改成自己的图
im = im.astype('float32')

result = model.predict(im)


if model.model_type == "classifier":
    print(result)

if model.model_type == "detector":
    pdx.det.visualize(im, result, threshold=0.5, save_dir='./')

if model.model_type == "segmenter":
    pdx.seg.visualize(im, result, weight=0.0, save_dir='./')
