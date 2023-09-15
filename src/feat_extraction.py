import mahotas
import pandas
import numpy
import glob
import cv2

labels = ['Healthy', 'Powdery_Mildew']

features_list = list()

for label in labels:
    
    path = './images/' + label

    images_list = glob.glob(path + '/*.jpg')

    for image_path in images_list:

        img = cv2.imread(image_path, 0)

        if img.shape[-1] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        img_gray = cv2.resize(img_gray, (256, 256))

        label = 1 if 'Healthy' in image_path else 0

        features_img = mahotas.features.haralick(img, compute_14th_feature = True, return_mean = True)

        features_img = numpy.append(features_img, label)

        features_list.append(features_img)

features_names = mahotas.features.texture.haralick_labels

features_names = numpy.append(features_names, 'Label')

df = pandas.DataFrame(data = features_list, columns = features_names)

df.to_csv('./etc/features.csv', index = False, sep = ';')

    

