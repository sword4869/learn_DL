```python
# deprecate: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
torchvision.models.resnet50(pretrained=True)

# deprecated 
torchvision.models.resnet50(weights=False)

# Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.
torchvision.models.resnet50(weights=None)

############# current
torchvision.models.vgg(weights=VGG16_Weights.DEFAULT)
```