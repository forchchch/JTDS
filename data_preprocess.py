import json
import os

root = './CUB_200_2011'
image_dictionary = {}
id_image_path = os.path.join(root,'images.txt')
######id: image_location, category_label, auxiliary label
with open(id_image_path,'r') as f:
    for line in f.readlines():
        info = line.split()
        image_dictionary[info[0]] = []
        image_dictionary[info[0]].append(info[1])

image_label_path = os.path.join(root,'image_class_labels.txt')
with open(image_label_path,'r') as f1:
    for line in f1.readlines():
        info = line.split()
        image_dictionary[info[0]].append( int(info[1])-1 )

image_attribute_path = os.path.join(root, 'attributes', 'image_attribute_labels.txt')
with open(image_attribute_path,'r') as f2:
    for line in f2.readlines():
        info = line.split()
        if len(image_dictionary[info[0]]) == 2:
            image_dictionary[info[0]].append([])
        image_dictionary[info[0]][2].append(int(info[2]))
outroot = os.path.join("./", 'preprocess_data')
with open( os.path.join(outroot,'image_dictionary.json'), 'w' ) as f3:
    json.dump(image_dictionary, f3)

###########begin training/validation/test/aux split
all_train_id_set = []
all_test_id_set = []
with open(os.path.join(root,'train_test_split.txt'),'r') as f4:
    for line in f4.readlines():
        info = line.split()
        if info[1] == '1':
            all_train_id_set.append(info[0])
        else:
            all_test_id_set.append(info[0])

print(all_test_id_set[0:12])
import random
random.shuffle(all_test_id_set)
print(all_test_id_set[0:12])

valid_id_set = all_test_id_set[0:2897]
test_id_set = all_test_id_set[2897:5794]
with open( os.path.join(outroot,'full_train_set.json'), 'w' ) as f:
    json.dump(all_train_id_set, f)

with open( os.path.join(outroot,'valid_set.json'), 'w' ) as f:
    json.dump(valid_id_set, f)

with open( os.path.join(outroot,'test_set.json'), 'w' ) as f:
    json.dump(test_id_set, f)

print("training length:",len(all_train_id_set))
print("valid length:",len(valid_id_set))
print("test length:",len(test_id_set))

#begin split the training set to the auxset and the real training set
import numpy as np
aux_shot = 1
aux_id_set = []
cate_recorder = np.zeros(200)
for img_id in all_train_id_set:
    cate_for_img = image_dictionary[img_id][1]
    if cate_recorder[cate_for_img]<aux_shot:
        aux_id_set.append(img_id)
        cate_recorder[cate_for_img] += 1
print(len(aux_id_set))

rest_train_set = []
for img_id in all_train_id_set:
    if img_id not in aux_id_set:
        rest_train_set.append(img_id)
print(len(rest_train_set))

with open( os.path.join(outroot,'aux_set.json'), 'w' ) as f:
    json.dump(aux_id_set, f)

with open( os.path.join(outroot,'rest_train_set.json'), 'w' ) as f:
    json.dump(rest_train_set, f)



