项目参考何凯明团队研发的Mask R-CNN目标检测网络：https://github.com/matterport/Mask_RCNN
用labelme标注工具完成数据集的制作，标记每张图片将产生json文档，我们按照下面的步骤就可以生成dataset：
1. 如果是将单个json文件生成数据集，则运行命令：
 \anaconda安装目录\envs\labelme\Scripts\label_json_to_dataset.exe   \保存xxx.json文件目录\xxx.json 
会成生成五个文件 img.png，info.yaml，label.png，label_names.txt，label_viz.png.其中label.png即是我们要的mask了.
![img.png](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/image/0000.png)![label.png](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/mask/0000.png)![label_viz.png](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/viz/0000.png)
2. 将批量的json文件生成数据集，编写python脚本如下：

import os

path = 'E:\\label_data\\'  # path是存放json和img的文件路径，需要将json和原始img文件放在一个文件夹下

json_file = os.listdir(path)

for file in json_file:

    if 'json' in file:
    
        os.system("python D:\\Anaconda3\\envs\\labelme\\Scripts\\labelme_json_to_dataset.exe %s"%(path + file))



数据集制作完毕运行[shapes_train.py](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/shapes_train.py)开始训练模型，开始先训练Mask R-CNN的‘head’网络部分，如果硬件条件允许的话后期之后可以finetune整个网络

最后运行[shapes_test.py](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/shapes_test.py)测试单张图片的识别效果

识别效果如下：
![test_image](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/test_images/1.png)
![test_image](https://github.com/brucezhcw/Neural-Network/blob/master/Mask%20RCNN/test_images/5.png)
