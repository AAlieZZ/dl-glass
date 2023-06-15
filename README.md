# dl-glass
通过卷积神经网络训练深度学习模型进行玻璃缺陷检测

## Getting Started
解压数据集
```
cd GlassCoverDefectDataset
7z x GlassCover_datset.zip
```
修改路径
```
sed -i 's/D:\\SSD_datset\\JPEGImages\\/\/path\/to\/dl-glass\/GlassCoverDefectDataset\/GlassCover_datset\/JPEGImages\//' GlassCover_datset/Annotations/*.xml
sed -i 's/D:\\yolov3-tf2-master\\SSD_datset\\JPEGImages\\/\/path\/to\/dl-glass\/GlassCoverDefectDataset\/GlassCover_datset\/JPEGImages\//' GlassCover_datset/Annotations/*.xml
```

## Build
```
cargo build --release
```

## Start
```
./dl-glass
```
