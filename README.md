![Alt text](https://www.deepblueai.com/usr/deepblue/v3/images/logo.png "DeepBlue")  
[![license](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://github.com/DeepBlueAI/AutoSmart/blob/master/LICENSE)

## Introduction

The 1st place solution for [AutoNLP 2019](https://www.4paradigm.com/competition/autoNLP2019). 

## Usage

Download  the competition's [starting kit](https://github.com/mortal123/autonlp_starting_kit) and run

```
python run_local_test.py -dataset_dir=./AutoDL_sample_data/DEMO -code_dir=./AutoDL_sample_code_submission
```

You can change the argument `dataset_dir` to other datasets, and change the argument `dataset_dir` to the directory containing this code (`model.py`).

And we use the embedding model provided by the competition, which is saved in `/app/embedding`.

The file `ac.cpython-36m-x86_64-linux-gnu.so` is compiled by Cython, and its source code is `ac.pyx` .

### Dataset

This challenge focuses on the problem of **multi-class text categorization** collected from real-world businesses. The datasets consist of content file, label file and meta file, where content file and label file are split into train parts and test parts:

\- **Content file ({train, test}.data)** contains the content of the instances. Each row in the content file represents the content of an instance.

\- **Label file ({train, dataset_name}.solution)** consists of the labels of the instances in one-hot format. Note that each of its lines corresponds to the corresponding line number in the content file.

\- **Meta file (meta.json)** is a json file consisted of the meta information about the dataset, including language, train instance number, test instance number, category number.

The following figure illustrates the form of the datasets:

![img](https://www.4paradigm.com/images/pc/autonlp/data1.png)



# Contact Us

DeepBlueAI: [1229991666@qq.com](mailto:1229991666@qq.com)