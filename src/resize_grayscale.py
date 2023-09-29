import cv2
import os

input_folder = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/Healthy/'

output_folder = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/image test/Healthy_resized_gray/'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        new_size = (256, 256)
        resized_image = cv2.resize(image, new_size)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        output_path = os.path.join(output_folder, 'resized_gray_' + filename)
        cv2.imwrite(output_path, gray_image)
