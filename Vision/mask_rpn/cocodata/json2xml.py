# coco2voc.py

# pip install pycocotools
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
 
#json文件路径和用于存放xml文件的路径
anno = '/disk1/ybh/datasets/COCO2017/annotations/instances_train2017.json'
xml_dir = '/disk1/ybh/datasets/COCO20172VOC/VOCdevkit/COCO2017/Annotations/'

coco = COCO(anno)  # 读文件
cats = coco.loadCats(coco.getCatIds())  # 这里loadCats就是coco提供的接口，获取类别
    
# Create anno dir
dttm = time.strftime("%Y%m%d%H%M%S", time.localtime())

def trans_id(category_id):
    names = []
    namesid = []
    for i in range(0, len(cats)):
        names.append(cats[i]['name'])
        namesid.append(cats[i]['id'])
    index = namesid.index(category_id)
    return index
    

def convert(anno,xml_dir): 

    with open(anno, 'r') as load_f:
        f = json.load(load_f)
    
    imgs = f['images']  #json文件的img_id和图片对应关系 imgs列表表示多少张图
    
    cat = f['categories']
    df_cate = pd.DataFrame(f['categories'])                     # json中的类别
    df_cate_sort = df_cate.sort_values(["id"], ascending=True)  # 按照类别id排序
    categories = list(df_cate_sort['name'])                     # 获取所有类别名称
    print('categories = ', categories)
    df_anno = pd.DataFrame(f['annotations'])                    # json中的annotation
    
    for i in tqdm(range(len(imgs))):  # 大循环是images所有图片,Tqdm是可扩展的Python进度条，可以在长循环中添加一个进度提示信息
        xml_content = []
        file_name = imgs[i]['file_name']    # 通过img_id找到图片的信息
        height = imgs[i]['height']
        img_id = imgs[i]['id']
        width = imgs[i]['width']
        
        version =['"1.0"','"utf-8"'] 
    
        # xml文件添加属性
        xml_content.append("<?xml version=" + version[0] +" "+ "encoding="+ version[1] + "?>")
        xml_content.append("<annotation>")
        xml_content.append("    <filename>" + file_name + "</filename>")
        xml_content.append("    <size>")
        xml_content.append("        <width>" + str(width) + "</width>")
        xml_content.append("        <height>" + str(height) + "</height>")
        xml_content.append("        <depth>"+ "3" + "</depth>")
        xml_content.append("    </size>")
    
        # 通过img_id找到annotations
        annos = df_anno[df_anno["image_id"].isin([img_id])]  # (2,8)表示一张图有两个框
    
        for index, row in annos.iterrows():  # 一张图的所有annotation信息
            bbox = row["bbox"]
            category_id = row["category_id"]
            cate_name = categories[trans_id(category_id)]
    
            # add new object
            xml_content.append("    <object>")
            xml_content.append("        <name>" + cate_name + "</name>")
            xml_content.append("        <truncated>0</truncated>")
            xml_content.append("        <difficult>0</difficult>")
            xml_content.append("        <bndbox>")
            xml_content.append("            <xmin>" + str(int(bbox[0])) + "</xmin>")
            xml_content.append("            <ymin>" + str(int(bbox[1])) + "</ymin>")
            xml_content.append("            <xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
            xml_content.append("            <ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
            xml_content.append("        </bndbox>")
            xml_content.append("    </object>")
        xml_content.append("</annotation>")
    
        x = xml_content
        xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]
        ### list存入文件
        #xml_path = os.path.join(xml_dir, file_name.replace('.xml', '.jpg'))
        xml_path = os.path.join(xml_dir, file_name.split('j')[0]+'xml')
        print(xml_path)
        with open(xml_path, 'w+', encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:] = []

if __name__ == '__main__':
    convert(anno,xml_dir)
