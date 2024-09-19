from pycocotools.coco import COCO
import os
import json
import cv2
import numpy as np
save_len=4
def get_new_filename(name):
    num,_=name.split('.')
    num=str(int(num))
    extra_num=save_len-len(num)
    new_name=extra_num*'0'+num
    return new_name
def get_new_maskname(name):
    num,_=name.split('_')
    num=str(int(num))
    extra_num=save_len-len(num)
    new_name = extra_num * '0' + num
    return new_name
def get_new_mask(old_mask_path,new_mask_path):
    img = cv2.imread(old_mask_path)
    # max_img=np.max(img,axis=0)
    # max_img=np.max(max_img,axis=0)
    # print(max_img)
    new_img = np.zeros_like(img)
    new_img[..., 0:1] = (img[..., 2:3] >240) * 255
    new_img[..., 1:2] = ((img[..., 1:2] !=108) * (img[...,1:2]>0)) * 255
    if new_img.shape[0]!=640 or new_img.shape[1]!=640:
        new_img=cv2.resize(new_img,(640,640))
    cv2.imwrite(new_mask_path, new_img)
def change_json(json_name='train'):
    save_dir= 'new_anns'
    main_dir= 'annotations'
    json_path=os.path.join(main_dir,json_name+'.json')
    coco=COCO(json_path)
    target_index=coco.getCatIds(['ship'])[0]
    imgIds=coco.getImgIds(catIds=[target_index])
    output_json_dict={
        "images":[],
        "type":"instances",
        "annotations":[],
        "categories":[]
    }
    save_dir_path=os.path.join(main_dir,save_dir)
    os.makedirs(save_dir_path,exist_ok=True)
    save_json_path=os.path.join(save_dir_path,json_path)
    bnd_id=1
    for i in imgIds:
        img_index=coco.loadImgs(i)[0]
        id=img_index['id']
        width=img_index['width']
        height=img_index['height']
        file_name=img_index['file_name']
        file_name=get_new_filename(file_name)
        image_info={
            'file_name':file_name+'.jpg',
            'height':height,
            'width':width,
            'id':id
        }
        output_json_dict['images'].append(image_info)
        anns=coco.imgToAnns[id]
        for ann in anns:
            ann.update({'image_id':id,'id':bnd_id,"category_id":0})
            bnd_id=bnd_id+1
            output_json_dict['annotations'].append(ann)
    for catid in coco.getCatIds():
        print(json_name)
        old_category=coco.loadCats(catid)[0]
        category_info = {'supercategory': 'none', 'id': old_category['id']-1, 'name': old_category['name']}
        output_json_dict['categories'].append(category_info)
        break
    with open(save_json_path,'w') as f:
        output_json=json.dumps(output_json_dict)
        f.write(output_json)
if __name__=='__main__':
    for name in ['train','test','val','trainval']:
        change_json(name)