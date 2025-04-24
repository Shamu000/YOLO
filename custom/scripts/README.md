###  split_annotations

```
将标注工具标完之后混在一个文件夹中的标签和图片分离开
```

### convert_format

```
标签格式转化，现在应该用不到了
```

### validate_labels

```
检查标签是否满足归一化条件等，不满足就删除标签
```

### remove_invalid_entries

```
检查传入的图片和标签文件夹下面的标签和文件是否一一对应，不对应就同时删除图片和标签
```

### split_dataset

```
在划分好images和labels的基础上，进一步划分为train和val
```

### visualize_boxes

```
将对应标签的数据画到图片上
```

### convert_PNG

```
将制定路径下的所有图片转换为PNG格式，适用于从工业相机拍摄的图片处理，后续用于标注工具中
```

### count_labels

```
统计指定路径下的所有文件夹里面的labels/train和labels/val里面的标签，并分别进行每个数据集类别总数和所有数据集类别总数汇总
```
## count_one_dataset

```
统计指定数据集下的所有文件夹里面的labels/train和labels/val里面的标签，并分别进行每个数据集类别总数和所有数据集类别总数汇总
```
