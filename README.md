# 轻量级文字识别技术创新大赛第9名方案

## 项目描述
模型总大小9.6M A榜精度79.96%，B榜精度79.43%  
模型结构：mobilenet + BiLSTM + transformer*2 + linear*2  
模型:训练模型提取码：

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
  ### 训练步骤
    1、数据准备  
       将训练集随机分为90%的训练集和10%的验证集；
       使用
    2、
    
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：此处由项目作者进行撰写使用方式。

最终的提交代码训练步骤：
