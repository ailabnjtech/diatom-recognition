import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import os.path
from numpy import average, linalg, dot

from classification import Classification, _preprocess_input
from utils.utils import letterbox_image
from PIL import ImageDraw, Image
import numpy as np
import os
import cv2
import sys



class top5_Classification(Classification):
    def detect_image(self, image):
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # 图片预处理，归一化
        photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        photo = np.transpose(photo,(0,3,1,2))

        with torch.no_grad():
            photo = Variable(torch.from_numpy(photo).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argsort(preds)[::-1]  # 输出元素从大到小排列的对应的索引
        arg_pred_top5 = arg_pred[:5]
        class_name0 = self.class_names[arg_pred_top5[0]] # 输出的是预测的名称
        class_name1 = self.class_names[arg_pred_top5[1]] # 输出的是预测的名称
        class_name2 = self.class_names[arg_pred_top5[2]] # 输出的是预测的名称
        class_name3 = self.class_names[arg_pred_top5[3]] # 输出的是预测的名称
        class_name4 = self.class_names[arg_pred_top5[4]] # 输出的是预测的名称


        probability = np.sort(preds)[::-1]  # 最大可能性的类别的可能性大小
        probability_top5=probability[:5]
        # # class_name = self.class_names[np.argsort(preds)[::-1]] # 最大可能性的类别所对应的名称
        return arg_pred_top5,class_name0,class_name1,class_name2,class_name3,class_name4, probability_top5

        # class_name = self.class_names[np.argmax(preds)] # 输出的是预测的名称
        # arg_pred_top5 = class_name[:5]
        # return arg_pred_top5




# fw = open(r"C:\Users\psy\Desktop\analysis\confidence/top1就预测正确的top5.csv", 'w')

def evaluteTop5(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].replace('\n', '')
        x = Image.open(annotation_path)
        y = int(line.split(';')[0]) # 图片本来所属的类别编号
        s = os.path.basename(os.path.dirname(line.split(';')[1]))  # 图片本来所属的类别名

        pred = classfication.detect_image(x)[0]
        name0 = classfication.detect_image(x)[1]
        name1 = classfication.detect_image(x)[2]
        name2 = classfication.detect_image(x)[3]
        name3 = classfication.detect_image(x)[4]
        name4 = classfication.detect_image(x)[5]
        pro = classfication.detect_image(x)[6]
        if pro[0]<0.3:
            similaritylist = []

            def get_thumbnail(image, size=(224, 224), greyscale=False):
                image = image.resize(size, Image.ANTIALIAS)
                if greyscale:
                    image = image.convert('L')
                return image

            def image_similarity_vectors_via_numpy(image1, image2):
                image1 = get_thumbnail(image1)
                image2 = get_thumbnail(image2)
                images = [image1, image2]
                vectors = []
                norms = []
                for image in images:
                    vector = []
                    for pixel_tuple in image.getdata():
                        vector.append(average(pixel_tuple))
                    vectors.append(vector)
                    norms.append(linalg.norm(vector, 2))
                a, b = vectors
                a_norm, b_norm = norms
                res = dot(a / a_norm, b / b_norm)
                return res

            image1 = Image.open(annotation_path)
            category = [name0,name1,name2,name3,name4]
            for j in range(5):

                path = r"./datasets/train resize224" + '/' + category[j]  # 可以变动的类别
                files = os.listdir(path)
                image_number = len(files)
                global all
                all = 0
                for i in files:
                    if os.path.splitext(i)[1].lower() == '.jpg' or '.png' or '.jpeg':
                        fileName = os.path.join(path, i)
                        image2 = Image.open(fileName)
                        cosin = image_similarity_vectors_via_numpy(image1, image2)
                        all += cosin
                averagecosin = all / image_number
                similaritylist.append(averagecosin)

                # print(all)
                # print(averagecosin)
            # prenumber = similaritylist.index(max(similaritylist))
            # print(prenumber)
            # print(similaritylist)

            prenumber = similaritylist.index(max(similaritylist))
            prenew =pred[prenumber]
            similaritylist.clear()
        else:
            prenew=pred[0]

        correct += y == prenew
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    return correct / total



classfication = top5_Classification()
with open(r"./cls_test.txt","r") as f:
    lines = f.readlines()
top5 = evaluteTop5(classfication, lines)
print("top-5 accuracy = %.2f%%" % (top5*100))

