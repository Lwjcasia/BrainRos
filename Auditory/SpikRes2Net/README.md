
# 基于Pytorch实现的声音分类系统

# 前言

本项目是基于脉冲神经网络的声音分类项目，旨在实现对各种环境声音、动物叫声和语种的识别。

# 使用准备

 - Anaconda 3
 - Python 3.11
 - Pytorch 2.0.1
 - Windows 11 or Ubuntu 22.04

## 安装环境

 - 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

```shell
pip install .
```

## 准备数据

生成数据列表，用于下一步的读取需要，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒以上，如 `dataset/audio/鸟叫声/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

以Urbansound8K为例，Urbansound8K是目前应用较为广泛的用于自动城市环境声分类研究的公共数据集，包含10个分类：空调声、汽车鸣笛声、儿童玩耍声、狗叫声、钻孔声、引擎空转声、枪声、手提钻、警笛声和街道音乐声。数据集下载地址：[UrbanSound8K.tar.gz](https://aistudio.baidu.com/aistudio/datasetdetail/36625)。以下是针对Urbansound8K生成数据列表的函数。如果读者想使用该数据集，请下载并解压到 `dataset`目录下，把生成数据列表代码改为以下代码。

执行`create_data.py`即可生成数据列表，里面提供了生成多种数据集列表方式，具体看代码。
```shell
python create_data.py
```

生成的列表是长这样的，前面是音频的路径，后面是该音频对应的标签，从0开始，路径和标签之间用`\t`隔开。
```shell
dataset/UrbanSound8K/audio/fold2/104817-4-0-2.wav	4
dataset/UrbanSound8K/audio/fold9/105029-7-2-5.wav	7
dataset/UrbanSound8K/audio/fold3/107228-5-0-0.wav	5
dataset/UrbanSound8K/audio/fold4/109711-3-2-4.wav	3
```

# 修改预处理方法（可选）

配置文件中默认使用的是Fbank预处理方法，如果要使用其他预处理方法，可以修改配置文件中的安装下面方式修改，具体的值可以根据自己情况修改。如果不清楚如何设置参数，可以直接删除该部分，直接使用默认值。

```yaml
# 数据预处理参数
preprocess_conf:
  # 是否使用HF上的Wav2Vec2类似模型提取音频特征
  use_hf_model: False
  # 音频预处理方法，也可以叫特征提取方法
  # 当use_hf_model为False时，支持：MelSpectrogram、Spectrogram、MFCC、Fbank
  # 当use_hf_model为True时，指定的是HuggingFace的模型或者本地路径，比如facebook/w2v-bert-2.0或者./feature_models/w2v-bert-2.0
  feature_method: 'Fbank'
  # 当use_hf_model为False时，设置API参数，更参数查看对应API，不清楚的可以直接删除该部分，直接使用默认值。
  # 当use_hf_model为True时，可以设置参数use_gpu，指定是否使用GPU提取特征
  method_args:
    sample_frequency: 16000
    num_mel_bins: 80
```

# 提取特征（可选）

在训练过程中，首先是要读取音频数据，然后提取特征，最后再进行训练。其中读取音频数据、提取特征也是比较消耗时间的，所以我们可以选择提前提取好取特征，训练模型的是就可以直接加载提取好的特征，这样训练速度会更快。这个提取特征是可选择，如果没有提取好的特征，训练模型的时候就会从读取音频数据，然后提取特征开始。提取特征步骤如下：

1. 执行`extract_features.py`，提取特征，特征会保存在`dataset/features`目录下，并生成新的数据列表`train_list_features.txt`和`test_list_features.txt`。

```shell
python extract_features.py --configs=configs/sres2net.yml --save_dir=dataset/features
```

2. 修改配置文件，将`dataset_conf.train_list`和`dataset_conf.test_list`修改为`train_list_features.txt`和`test_list_features.txt`。


## 训练

接着就可以开始训练模型了，创建 `train.py`。配置文件里面的参数一般不需要修改，但是这几个是需要根据自己实际的数据集进行调整的，首先最重要的就是分类大小`dataset_conf.num_class`，这个每个数据集的分类大小可能不一样，根据自己的实际情况设定。然后是`dataset_conf.batch_size`，如果是显存不够的话，可以减小这个参数。

```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```


**训练可视化：**

项目的根目录执行下面命令，并网页访问`http://localhost:8040/`，如果是服务器，需要修改`localhost`为服务器的IP地址。
```shell
visualdl --logdir=log --host=0.0.0.0
```

打开的网页如下：

<br/>
<div align="center">
<img src="docs/images/log.jpg" alt="混淆矩阵" width="600">
</div>



# 评估

执行下面命令执行评估。

```shell
python eval.py --configs=configs/bi_lstm.yml
```


# 预测

在训练结束之后，我们得到了一个模型参数文件，我们使用这个模型预测音频。

```shell
python infer.py --audio_path=dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav
```

# 其他功能

 - 为了方便读取录制数据和制作数据集，这里提供了录音程序`record_audio.py`，这个用于录制音频，录制的音频采样率为16000，单通道，16bit。

```shell
python record_audio.py
```

 - `infer_record.py`这个程序是用来不断进行录音识别，我们可以大致理解为这个程序在实时录音识别。通过这个应该我们可以做一些比较有趣的事情，比如把麦克风放在小鸟经常来的地方，通过实时录音识别，一旦识别到有鸟叫的声音，如果你的数据集足够强大，有每种鸟叫的声音数据集，这样你还能准确识别是那种鸟叫。如果识别到目标鸟类，就启动程序，例如拍照等等。

```shell
python infer_record.py --record_seconds=3
```


# 参考资料

1. https://github.com/yeyupiaoling/AudioClassification-Pytorch
