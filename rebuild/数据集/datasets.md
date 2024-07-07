- [intro](#intro)
  - [缓存](#缓存)
- [1. 下载hub上的dataset](#1-下载hub上的dataset)
  - [1.1. hub的name](#11-hub的name)
  - [1.3. 下载到本地](#13-下载到本地)
  - [1.4. revision](#14-revision)
- [2. 本地数据集 imagefolder](#2-本地数据集-imagefolder)
  - [2.1. split等价](#21-split等价)
  - [2.2. imagefolder](#22-imagefolder)
  - [2.3. readme](#23-readme)
  - [2.4. data\_files](#24-data_files)
- [3. examples - transform](#3-examples---transform)


---

这库在老服务器上跑不行，
```
ImportError: /lib64/libc.so.6: version `GLIBC_2.25' not found (required by /public/home/hpc2204081200015/envs/nerf/lib/python3.10/site-packages/pyarrow/libparquet.so.1300)
```

## intro

why `datasets.load_dataset()`?

In distributed training, the `load_dataset` function guarantees that only one local process can concurrently download the dataset.

### 缓存

    
缓存目录是一个本地文件夹，用于存储下载和生成的数据集。使用缓存目录可以避免每次使用数据集时都重新下载或处理整个数据集。

`cache_dir` 的默认缓存目录是 `'~/.cache/huggingface/datasets'`

修改缓存目录的位置：

- 您可以通过设置环境变量`HF_DATASETS_CACHE`来更改缓存目录的位置。例如：

    ```python
    $ export HF_DATASETS_CACHE="/path/to/another/directory"
    ```
- 您也可以在调用`load_dataset()`时，通过`cache_dir`参数来指定缓存目录的路径。例如：
    ```python
    dataset = load_dataset(
        "lambdalabs/pokemon-blip-captions",
        cache_dir="/path/to/another/directory"
    )
    # /path/to/another/directory
    # └── pokemon-blip-captions
    #     └── lambdalabs--pokemon-blip-captions-9398e610fc2f9632
    #         └── 0.0.0
    #             └── e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990
    #                 ├── dataset_info.json
    #                 └── pokemon-blip-captions-train.arrow
    ```

## 1. 下载hub上的dataset
### 1.1. hub的name

```python
dataset = load_dataset("lambdalabs/pokemon-blip-captions")
# ~/.cache/huggingface/datasets
# └── pokemon-blip-captions
#     └── lambdalabs--pokemon-blip-captions-9398e610fc2f9632
#         └── 0.0.0
#             └── e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990
#                 ├── dataset_info.json
#                 └── pokemon-blip-captions-train.arrow
```
- 虽然其缓存目录只用构建一次，但是每次需要联网检查。


当缓存好之后就可以脱机使用缓存的数据集，而**不用联网**，比如使用默认的缓存位置`~/.cache/huggingface/datasets`。

但是这样每次都会在`cache_dir` 中重复缓存 `default-xxxxxxxxxxxxxxx`, 内容和`lambdalabs--pokemon-blip-captions-9398e610fc2f9632`完全一致。

```python
# 第一次在线下载
dataset = load_dataset("lambdalabs/pokemon-blip-captions")

# 之后脱机使用
dataset = load_dataset(
    "~/.cache/huggingface/datasets/pokemon-blip-captions"
)
# ~/.cache/huggingface/datasets/pokemon-blip-captions
# ├── default-d7ed22bbd57f2d3d
# ├── default-fa65dad88655f9f0
# └── lambdalabs--pokemon-blip-captions-9398e610fc2f9632
```
### 1.3. 下载到本地

```bash
git clone https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
```
```bash
pokemon-blip-captions
├── README.md
├── dataset_infos.json
└── data
    └── train-00000-of-00001-566cc9b19d7203f8.parquet
```
只用指定 path
```python
dataset = load_dataset('./pokemon-blip-captions')
```


### 1.4. revision

version based on Git tags, branches, or commits. 

```python
dataset = load_dataset(
  "lhoestq/custom_squad",
  revision="main"  # tag name, or branch name, or commit hash
)
```

## 2. 本地数据集 imagefolder


### 2.1. split等价
- train, training
- test, testing, eval, evaluation
- validation, valid, val, dev
### 2.2. imagefolder

```
pokemon/train/grass/bulbasaur.png
pokemon/train/fire/charmander.png
pokemon/train/water/squirtle.png

pokemon/test/grass/ivysaur.png
pokemon/test/fire/charmeleon.png
pokemon/test/water/wartortle.png
```
```python
dataset = load_dataset(
    "imagefolder", 
    data_dir="/path/to/pokemon"
)
# DatasetDict({
#     train: Dataset({
#         features: ['image'],
#         num_rows: 30000
#     }),
#     test: Dataset({
#         features: ['image'],
#         num_rows: 1000
#     })
# })

image = dataset['train'][0]['image']      # PIL
```


`cache_dir`用于存储**处理后**的数据集，而`data_dir`用于存储**原始的**数据集。

`cache_dir`可以提高加载速度，而`data_dir`可以节省下载流量。
### 2.3. readme
```
my_dataset_repository/
├── README.md
└── data/
    ├── training.csv
    ├── eval.csv
    └── valid.csv
```
在 `README.md`中还可以使用YAML来配置。

```
---
configs:
- config_name: default
  data_files:
  - split: train
    path: "data.csv"
  - split: test
    path: "holdout.csv"
---
```

### 2.4. data_files

`data_dir` is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.

A dataset without a loading script by default loads all the data into the train split. 

Use the `data_files` parameter to map data files to **splits** like train, validation and test:

```
my_dataset_repository
├── training.csv
├── eval.csv
└── valid.csv
```

```python
data_files = {
    "train": "train.csv", 
    "test": "test.csv"
}
dataset = load_dataset(
    "namespace/your_dataset_name", 
    data_files=data_files
)
```



```
pokemon/train/grass/bulbasaur.png
pokemon/train/fire/charmander.png
pokemon/train/water/squirtle.png

pokemon/test/grass/ivysaur.png
pokemon/test/fire/charmeleon.png
pokemon/test/water/wartortle.png
```
```python
data_files = {
    'train': os.path.join("/path/to/pokemon/train", "**"),
    'test': os.path.join("/path/to/pokemon/test", "**"),
}
dataset = load_dataset(
    "imagefolder",
    data_files=data_files
)
```

## 3. examples - transform

```python

def prepare_dataloader(image_size=32, batch_size=64):
    # dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    dataset = load_dataset('./dataset', split="train")

    # Or load images from a local folder
    # dataset = load_dataset("imagefolder", data_dir="path/to/folder")

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB"))
                  for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader


def train(train_dataloader, model, optimizer, loss_function, scheduler, num_epoch=30):
    model.train()
    for epoch in range(num_epoch):
        for batch in tqdm(train_dataloader):
            clean_images = batch["images"].to(device)
```