
## MaxRetryError
```
(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: 
```
`pip install --upgrade requests`

## LocalEntryNotFoundError

```
LocalEntryNotFoundError: Connection error, and we cannot find the requested files in the disk cache.
```

没用
- 清除 cache 文件夹. 默认在`~/.cache/huggingface/hub`.

只能一次次try, 然后就好了。

这次没有在Notebook中，而是写成py文件，结果一次就通过。是不是 notebook 文件的原因。

## SafetensorError: Error while deserializing header: HeaderTooLarge

下载出问题了，文件创建了，但是没有下全。

检查文件的大小，重新下载。