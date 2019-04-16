from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch
from torchvision import datasets, models, transforms

# 绘制混淆矩阵示例代码
# guess = [1, 0, 1]
# fact = [0, 1, 0]
# classes = list(set(fact))
# classes.sort()
# confusion = confusion_matrix(guess, fact)
# plt.imshow(confusion, cmap=plt.cm.Blues)
# indices = range(len(confusion))
# plt.xticks(indices, classes)
# plt.yticks(indices, classes)
# plt.colorbar()
# plt.xlabel('guess')
# plt.ylabel('fact')
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[first_index][second_index])
#
# plt.show()

# 全局变量
data_dir = "./DLdataset"
batch_size = 16
input_size = 224

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 65)
model_ft.load_state_dict(torch.load("/Users/tianyu/PycharmProjects/DL2/scrachResNet.pt", map_location='cpu'))

print(model_ft)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
print(image_datasets)

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'valid']}

guess = []
fact = []
# for labels in image_datasets['valid'].imgs:
#     # labels = labels.to(device)
#     fact.append(labels[1])
for inputs, labels in dataloaders_dict['valid']:
    # print(inputs)
    # pass
    # # inputs = inputs.to(device)
    # outputs = model_ft(inputs)
    # labels=labels[1].data.numpy().squeeze()
    for label in labels:
        label = label.data.numpy().tolist()
        fact.append(label)
        # print(type(label))

    outputs = torch.max(model_ft(inputs), 1)[1].data.numpy().tolist()
    for output in outputs:
        # output=output.astype(int)
        guess.append(output)
        # print(type(output),output)
    print("------Batch Complete------")

classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('guess')
plt.ylabel('fact')
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()

print("success!")
