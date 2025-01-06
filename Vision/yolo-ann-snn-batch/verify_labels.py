from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib import patches
import numpy as np
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class_path = 'data/coco.names'
class_list = load_classes(class_path)
img_path = 'data/train2014/images/COCO_train2014_000000000127.jpg'
img = np.array(Image.open(img_path))
H, W, C = img.shape
label_path = 'data/train2014/labels/COCO_train2014_000000000127.txt'
boxes = np.loadtxt(label_path, dtype=np.float).reshape(-1, 5)
# xywh to xxyy
boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * W
boxes[:, 2] = (boxes[:, 2] - boxes[:, 4] / 2) * H
boxes[:, 3] *= W
boxes[:, 4] *= H
fig = plt.figure()
ax = fig.subplots(1)
for box in boxes:
    bbox = patches.Rectangle((box[1], box[2]), box[3], box[4], linewidth=2,
                            edgecolor='r', facecolor="none")
    label = class_list[int(box[0])]
    # Add the bbox to the plot
    ax.add_patch(bbox)
    # Add label
    plt.text(box[1], box[2], s=label,
             color="white",
             verticalalignment="top",
             bbox={"color": 'g', "pad": 0},
            )
    ax.imshow(img)
plt.show()
