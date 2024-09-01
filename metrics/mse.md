
| Abbreviation | Full name | Mathematical expression |
|:------------:|:---------:|:-----------------------:|
| **L1 / mae** | Mean Absolute Error | $$\frac{1}{n} \sum \|y_{pred} – y_{gt}\|$$ |
| AbsRel | Absolute Relative Error | $$\frac{1}{n} \sum \frac{\|y_{pred} – y_{gt}\|}{y_{gt}}$$ |
| log mae | Mean Absolute Logarithmic Error | $$\frac{1}{n} \sum \|\log(y_{pred}) – log(y_{gt})\|$$ |
| imae | Inverse Mean Absolute Error | $$\frac{1}{n} \sum \|\frac{1}{y_{pred}} –\frac{1}{y_{gt}}\|$$ |
| **L2 / MSE** |  Mean Square Error | $${\frac{1}{n}\sum(y_{pred}- y_{gt})^2}$$ |
| RMSE | Root Mean Square Error | $$\sqrt{\frac{1}{n}\sum(y_{pred}- y_{gt})^2}$$ |
| log RMSE | Root Mean Square Logarithmic Error | $$\sqrt{\frac{1}{n}\sum(\log(y_pred)–\ log(y_gt))^2}$$ |
| iRMSE | Inverse Root Mean Square Error | $$\sqrt{\frac{1}{n}\sum(\frac {1 } {y_pred}-\ frac {1 } {y_gt})^2}$$ |
| SqRel | Square Relative Error | $$\frac {1} {n}\sum \frac {(y_{pred}-y_{gt})^2} {y_{gt}}$$ |