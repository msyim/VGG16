# VGG16

an implementation of VGG16 in pytorch.
VGG16 architecture is well depicted in the following image:

![](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)

image source : https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/


### modified (batch norm. added in each cnn layer)
```python
        # 1-1 conv layer
        tnn.Conv2d(3, 64, kernel_size=3, padding=1),
        tnn.BatchNorm2d(64),
        tnn.ReLU(),

        # 1-2 conv layer
        tnn.Conv2d(64, 64, kernel_size=3, padding=1),
        tnn.BatchNorm2d(64),
        tnn.ReLU(),

```
