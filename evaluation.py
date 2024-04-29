from __future__ import print_function, absolute_import, division
from collections import namedtuple
from cityscapesscripts.helpers.labels import *
import Helper.Helper_functions as hf



print("List of cityscapes labels:")
print("")
print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {:>14}".format('name', 'id', 'trainId', 'category',
                                                                              'categoryId', 'hasInstances',
                                                                              'color', 'ignoreInEval'))
print("    " + ('-' * 98))
counter = 0
for label in labels:
    #if label.ignoreInEval ==0:
    counter +=1
    print("     {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {}".format(label.name, label.id, label.trainId,
                                                                                  label.category, label.categoryId,
                                                                                  label.hasInstances,
                                                                                  str(label.color),  # Convert the tuple to a string
                                                                                  label.ignoreInEval,
                                                                                  ))
print(" ", '\n', counter)

label_dict = {}
for label in labels:
    if label.trainId != -1:
        if label.trainId not in label_dict:
            label_dict[label.trainId] = []
            label_dict[label.trainId].append([label.id, label.name, str(label.color)])
        else:
            label_dict[label.trainId].append([label.id, label.name, str(label.color)]) 
        #print(f'{label.id}, #{label.name}, Nr: {counter}')

city_label_color_map = []

for label in labels:
    if label.trainId != -1 and label.trainId not in city_label_color_map:
        print(f'{label.color}, #ID__{label.trainId}, {label.name}')
        city_label_color_map.append(label.trainId)

