要最大化的，就叫 Object function；要最小化的，就要 Loss function


---
Q: 损失函数是什么

A1: 损失函数就是预测值和真实值之间的差值。

A2：损失是样本的预测值和真实值之间的差值，用于计算损失的函数称为损失函数。损失函数是非负实值函数。


---

L1正则化，平均绝对误差有什么区别？
均方误差和L2？
L1正则化是L1 loss吗？

---



针对二元分类器



- 单独

    true false: 识别正确与否

    positive negative: 识别为A类，B类

- 组合
  
    true positive：识别为A类，确实是A类
    
    true negative: 识别为B类，确实是B类
    
    false positive(Type I error)：识别为A类，但其实是B类
    
    false negative(Type I error)：识别为B类，但其实是A类

- 公式
  
    $$p-tp=fn$$

    $$n-tn=fp$$

    $$\text {precision}=\dfrac{tp}{(tp+fp)}=\dfrac{tp}{p^{\prime}}$$

    $$\text {recall}=\dfrac{tp}{(tp+fn)}=\dfrac{tp}{p}$$

e.g.

真实：狗猫70/30
识别：分类是65/35，65张狗中60张（tp）是狗。但其实10张（fn）狗的没有识别到，这被识别为猫了。

![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062007803.png)  