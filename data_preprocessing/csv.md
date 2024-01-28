```python
import csv
def save_pred(file, column_names, preds):
    ''' Save predictions to specified file '''
    print(f'Saving results to {file}')
    with open(file, 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        # writer.writerow(['id', 'tested_positive'])
        writer.writerow(column_names)
        for i, p in enumerate(preds):
            writer.writerow([i, p])
```