from __future__ import print_function, absolute_import, division
from collections import namedtuple
from cityscapesscripts.helpers.labels import *



print("List of cityscapes labels:")
print("")
print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format('name', 'id', 'trainId', 'category',
                                                                              'categoryId', 'hasInstances',
                                                                              'color'))
print("    " + ('-' * 98))
counter = 0
for label in labels:
    if label.ignoreInEval ==0:
        counter +=1
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {}".format(label.name, label.id, label.trainId,
                                                                                  label.category, label.categoryId,
                                                                                  label.hasInstances,
                                                                                  label.color))
print(" ", '\n', counter)

print("Example usages:")

# Map from name to label
name = 'car'
id = name2label[name].id
print("ID of label '{name}': {id}".format(name=name, id=id))

# Map from ID to label
category = id2label[id].category
print("Category of label with ID '{id}': {category}".format(id=id, category=category))

# Map from trainID to label
trainId = 0
name = trainId2label[trainId].name
print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))