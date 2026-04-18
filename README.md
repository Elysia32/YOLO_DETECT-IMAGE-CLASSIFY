# YOLO_DETECT-IMAGE-CLASSIFY
软件主要由CLAUDE编写，yolo使用的代码思路来自CSDN大佬@落花不写码https://nuyoahinuhz.blog.csdn.net/?type=blog<br>
(如有侵权联系我删 3177919536@qq.com)<br>
使用软件自动进行yolo图像集划分以及生成训练的py文件<br>
detect文件夹内的main.py可以对模型进行测试，包括图片测试，视频测试，摄像头测试。<br>
train文件夹存放训练文件生成软件，main.py运行<br>
软件使用：<br>
我用labelimg打框，将图片路径和标签路径填完并执行可以自动识别并生成data.yaml<br>
测试集比例用来测试，0.1-0.2即可<br>
剩下的用来分配验证集和训练集，验证集也为0.1-0.2即可<br>
如果验证集为0.1，测试集为0.1，则训练集为(1-0.1)*(1-0.1)<br>

模型训练页面参数有对应解释，第一个可以不用yaml文件，改用pt文件
