# 轻量级文字识别技术创新大赛第9名方案

## 项目描述
模型总大小9.6M A榜精度79.96%，B榜精度79.43%  
模型结构：mobilenet + BiLSTM + transformer*2 + linear*2  
模型:[百度网盘](https://pan.baidu.com/s/1QulS3av8WXWv4k8XN7xTHw)    
密码: 0i8f

## 项目结构

  ### 数据增强
  使用iaa+运动模糊，参考[class DataAug:](https://github.com/crossLi/Ultra_light_OCR_No.9/blob/2cb2a04704ed798a9af554aa91ff69923f3aaf8b/ppocr/data/imaug/rec_img_aug.py#L32)
  ### 代码运行
  1、训练  
  
  ```
    sh trian.sh
  ```
  2、模型转换
  ```
    sh convert.sh
  ```
  3、预测
   ```
   sh predict.sh
   ```
  ### 训练策略
    1、数据准备  
       将训练集随机分为90%的训练集和10%的验证集； 
       将训练与验证数据转换为lmdb格式；  
    2、训练步骤 step1  
       使用resnet18 + 12层transformer + adadelta固定学习率0.001 + DataAug 进行训练3000个epoch
    3、训练步骤 step2 
       使用step1 pretrain训练模型
       将12层transformer减少至两层，其余参数不变，继续训练，大概训练1000个epoch
    4、训练步骤 step3  
       使用step2 pretrain训练模型
       将resnet18换成mobilenetV3，其余参数不变，训练500个epoch，将adadelta换成adam初始学习率为0.0001，同时去掉dataAug只保留原始的warp进行数据增强，再训练500个epoch
    5、训练步骤 step4  
       将全部数据转换为lmdb进行训练
       然后用step3进行pretrain，训练500个epoch,得到最后的模型转化为预测模型
    
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：此处由项目作者进行撰写使用方式。

最终的提交代码训练步骤：
