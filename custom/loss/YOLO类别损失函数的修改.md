# YOLO类别损失函数的修改

`ultralytics/utils/loss.py`

## 求label

无论是`VariFocal Loss` 还是` Focal Loss`，都有一个参数`label`，`label` 通常是二进制的（0 或 1）

对于二分类问题，`label` 表示每个样本属于哪个类别

对于多分类问题，`label` 表示每个样本是否正确地分类



`v8PoseLoss`，多分类问题，`target_labels` 作为`VariFocal Loss` 和 `Focal Loss` 的参数`label`

```
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
```

`target_labels`需要和`target_scores`形状一样，所以先构建出来

`fg_mask`是一个[batchsize, 锚点个数,类别]

## VariFocalLoss的使用

### 1、使用官方的`VariFocalLoss`

求`label`参数以及对于`VariFocalLoss` 的调用

在`__init__`中：`self.varifocal_loss=VariFocalLoss()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.varifocal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

返回值：损失函数值的和

### 2、`VariFocalLoss2`

将原本`VariFocalLoss`的对结果先求平均再求和改成对结果求和

（1）类定义

```
class VarifocalLoss2(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        #默认参数alpha=0.75, gamma=2.0
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .sum()
            )
        return loss
```

（2）调用

在`__init__`中：`self.varifocal_loss=VariFocalLoss2()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.varifocal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 3、自己改写：`MyVariFocalLoss1`

将原本`VariFocalLoss`里的`F.binary_cross_entropy_with_logits`换成`nn.BCEWithLogitsLoss`，相当于在原装的`self.bce`外面套上`VariFocal Loss`，最后结果先平均再求和

（1）类定义

```
class MyVarifocalLoss1(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        #默认参数alpha=0.75, gamma=2.0
        """Computes varfocal loss."""
        bce=nn.BCEWithLogitsLoss(reduction="none")
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (bce(pred_score.float(), gt_score.float()) * weight)
                .mean(1)
                .sum()
            )
        return loss
```

（2）`MyVariFocalLoss1`的调用

在`__init__`中：`self.varifocal_loss=MyVariFocalLoss1()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.varifocal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 4、自己改写：`MyVariFocalLoss2`

将原本`VariFocal Loss`里的`F.binary_cross_entropy_with_logits`换成`nn.BCEWithLogitsLoss`，相当于在原装的`self.bce`外面套上`VariFocal Loss`，最后结果求和

（1）类定义

```
class MyVarifocalLoss2(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        #默认参数alpha=0.75, gamma=2.0
        """Computes varfocal loss."""
        bce=nn.BCEWithLogitsLoss(reduction="none")
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (bce(pred_score.float(), gt_score.float()) * weight)
                .mean(1)
                .sum()
            )
        return loss
```

（2）`MyVariFocalLoss2`的调用

在`__init__`中：`self.varifocal_loss=MyVariFocalLoss2()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.varifocal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 5 效果

（1）官方的`VariFocal Loss`：

cls_loss初始值都是0.01数量级甚至更小，P在0.01以下波动，map在0.005左右波动

==（2）`VariFocalLoss2`：==

初始cls_loss 1.592 ，200e结果P=0.755     R=  0.58    map=  0.681    map95=  0.681

（3）自己写的`VariFocal Loss1`：

cls_loss初始值都是0.01数量级甚至更小，P在0.03上下波动，map在0.05上下波动

==（4）自己写的`VariFocal Loss2`：==

初始cls_loss 1.63 ，200e结果：P=0.712    R=  0.636    map50=  0.698    map95=  0.698

==（是因为Lr太小导致局部最优吗）==

## Focal Loss的使用

### 1 官方`FocalLoss`的使用

求`label`参数以及对于`FocalLoss` 的调用

在`__init__`中：`self.focal_loss=FocalLoss()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.focal_loss(pred_scores, target_labels) /target_scores_sum
```

返回值：损失函数值的和

### 2 `FocalLoss2`

将官方`FocalLoss` 对于结果的先平均再求和，改成对于结果求和

（1）类定义

```
class FocalLoss2(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        #默认 gamma=1.5, alpha=0.25
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.sum()
```

（2）调用

在`__init__`中：`self.focal_loss=FocalLoss2()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.focal_loss(pred_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 3 自己改写：`MyFocalLoss1`   

将原本`Focal Loss`里的`F.binary_cross_entropy_with_logits`换成`nn.BCEWithLogitsLoss`，相当于在原装的`self.bce`外面套上`Focal Loss`,**结果先平均后求和**

（1）类定义：

```
class MyFocalLoss1(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, target, label, gamma=1.5, alpha=0.25):
        #默认 gamma=1.5, alpha=0.25
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        bce=nn.BCEWithLogitsLoss(reduction="none")
        loss = bce(pred, target)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()
```

（2）`MyVariFocalLoss1`的调用

在`__init__`中：`self.focal_loss=MyFocalLoss1()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.focal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 4  自己改写：`MyFocalLoss2`    

将原本`Focal Loss`里的`F.binary_cross_entropy_with_logits`换成`nn.BCEWithLogitsLoss`，相当于在原装的`self.bce`外面套上`Focal Loss`，对于最终的结果**求和**

（1）类定义：

```
class MyFocalLoss2(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, target, label, gamma=1.5, alpha=0.25):
        #默认 gamma=1.5, alpha=0.25
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        bce=nn.BCEWithLogitsLoss(reduction="none")
        loss = bce(pred, target)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.sum()
```

（2）`MyVariFocalLoss2`的调用

在`__init__`中：`self.focal_loss=MyFocalLoss2()`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.focal_loss(pred_scores, target_scores, target_labels) /target_scores_sum
```

（3）返回值：损失函数值的和

### 5 参考博客：`FocalLoss3`

（1）定义

```
class FocalLoss3(nn.Module):
    """用在代替原本的BCEcls（分类损失）和BCEobj（置信度损失）
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    论文: https://arxiv.org/abs/1708.02002
    https://blog.csdn.net/qq_38253797/article/details/116292496
    TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    """
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()  定义为多分类交叉熵损失函数
        self.gamma = gamma # 参数gamma  用于削弱简单样本对loss的贡献程度
        self.alpha = alpha # 参数alpha  用于平衡正负样本个数不均衡的问题
        self.reduction = loss_fcn.reduction  # self.reduction: 控制FocalLoss损失输出模式 sum/mean/none  默认是Mean
        # focalloss中的BCE函数的reduction='None'  BCE不使用Sum或者Mean 
        # 需要将Focal loss应用于每一个样本之中
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        # 正常BCE的loss:   loss = -log(p_t)
        loss = self.loss_fcn(pred, true) 
        # p_t = flow.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma # 这里代表Focal loss中的指数项
        # 返回最终的loss=BCE * 两个参数  (看看公式就行了 和公式一模一样)
        loss = loss * alpha_factor * modulating_factor
        # 最后选择focalloss返回的类型 默认是mean
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

```

（2）调用

在`__init__`中：`self.focal_loss=FocalLoss3(nn.BCEWithLogitsLoss(reduction='none'))`

在`__call__`中

```
target_scores_sum = max(target_scores.sum(), 1)
target_labels = torch.zeros_like(target_scores)  # (b, h*w, 1)
target_labels[fg_mask] = 1.0 
loss[3] = self.focal_loss(pred_scores, target_scores.to(dtype)).sum() /target_scores_sum
```

（3）实例函数返回值取决于`nn.BCEWithLogitsLoss`参数

`reduction='none'`   返回张量

`reduction='mean'`   返回均值

`reduction='sum'`   返回和

### 6  效果

（1）官方  (我在用VariFocalLoss和FocalLoss没有除以sum)

1）cls_loss 初始值1.618

2） P稳定在0.035上下，map稳定在0.04上下   ，无法正常训练

（2）FocalLoss2（如果官方版最后的先平均再求和改成直接求和）：

初始cls_loss 387.4  ，稳定时P=0.0371     R= 0.516   map=  0.0532    map95= 0.0518

（3）自己写的MyFocalLoss1，最后先平均再求和

(没有除以sum)  cls_loss  0.01数量级,P和map缓慢的增加，最后P稳定在0.55,map稳定在0.17

如果除以sum，差不多会这样：0.00137     0.0653    0.00398    0.00383

==（4）MyFocalLoss2（ 最后求和）==  最优

初始cls_loss约为4.6,可以正常训练，200e结果0.905       0.85      0.939      0.938

==（5）FocalLoss3博客版本==

初始cls_loss约为0.9,可以正常训练，200e结果0.866      0.764      0.812      0.811