# Speech Emotion Recognition using Deep Learning and Discrete Wavelet Transform

Repository dedicated to the developed solution for the end-of-course work of the University of São Paulo's [Master in Business Administration (MBA) in Artificial Intelligence and Big Data](https://mba.iabigdata.icmc.usp.br/) program. The solution's main focus is to tackle the Speech Emotion Recognition task and is composed of a Convolutional Neural Network based on [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) trained using Mel Spectrogram and Discrete Wavelet Transform.

How to cite the paper (soon):

...

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Paper's Results](#papers-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install this package, first, you should clone the repository to the directory of your choice using the following command:
```bash
git clone https://github.com/rafaelgreca/ser-wavelet.git
```

P.S.: if you are interested in modifying the code as you desire, it's better to fork this repository and then clone your repository.

### Installing using Virtual Environment

Finally, you need to create a conda environment (or any other virtual environment solution, like virtualenv) and install the requirements. This can be done using `pip` with the following command:
```bash
conda create --name ser python=3.8.10 && conda activate ser
pip install -r requirements.txt
```

### Installing using Docker (Recommended)

Build the image using the following command:
```bash
docker build -t ser-wavelet -f Dockerfile .
```

## Getting Started

After installing the dependencies you are ready to run the code. But first, let's understand the code and its input arguments, shall we?

### Download the Datasets

Before continuing, to the code work properly you need to download the datasets correctly. If you install using other sources, the code might not work due to different folder organization and different file/folder names. Download the datasets using the links below:

- [CORAA](https://github.com/rmarcacini/ser-coraa-pt-br);
- [RAVDESS](https://zenodo.org/record/1188976);
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/);
- [EMODB](http://www.emodb.bilderbar.info).

### Directory Structure

```bash
./
├── config/
│   ├── mode_1.json
│   └── mode_2.json
├── examples/
│   ├── data_augmentation/
│   │   ├── audioaugment.json
│   │   ├── cutmix.json
│   │   ├── denoiser.json
│   │   ├── mixup.json
│   │   ├── multiple_techniques.json
│   │   ├── multiple_techniques_raw.json
│   │   ├── no_augment2.json
│   │   ├── no_augment.json
│   │   ├── specaugment.json
│   │   └── specmix.json
│   ├── feature/
│   │   ├── mel_spectrogram.json
│   │   └── mfcc.json
│   └── no_kfold.json
├── __init__.py
├── LICENSE
├── notebooks/
│   ├── analysis.ipynb
│   ├── eda.ipynb
│   ├── parsifal-analysis.ipynb
│   └── visualize-training.ipynb
├── README.md
├── src/
│   ├── data_augmentation.py
│   ├── dataset.py
│   ├── features.py
│   ├── __init__.py
│   ├── models/
│   │   ├── aghajani.py
│   │   ├── qiuqiangkong.py
│   │   └── utils.py
│   ├── processing.py
│   └── utils.py
├── requirements.txt
├── test.py
└── train.py
```

Explaining briefly the main folders and files:

- `config`: where the model and feature extraction configurations files (**MUST** be in a JSON format) that will be used as input arguments are saved on;
- `examples`: where the configuration files examples are saved (how to use different features, data augmentation techniques, and so on);
- `notebooks`: where the notebooks used to visualize the training/validation step, do the exploratory data analysis (EDA) and create the analysis used for the thesis writing are saved;
- `src`: where the core functions are implemented, such as the data augmentation, feature extraction, audio preprocessing steps, and the models/datasets creation;
- `test.py`: the main file responsible for inference of the model on new data;
- `train.py`: the main file responsible for training the model.

### Inputs Parameters

The following arguments **MUST** be inside the configuration JSON file (exactly as structured in the files inside the `config` folder).

- `input_path`: the dataset's folder path (**MUST** exists);
- `output_path`: the folder path where the extracted features will be saved (if the folder doesn't exist, then it will be created);
- `sample_rate`: the audio's sampling rate. E.g.: 8000;
- `dataset`: which dataset will be used ('coraa', 'savee', 'emodb' or 'ravdess'). **MUST** be written in lowercase;
- `mode`: which mode will be run ('mode\_1' or 'mode\_2'). Check the image below for more details. **MUST** be written in lowercase;
- `overwrite`: overwrite the extracted features or not;
- `to_mono`: convert the audios to mono or not;
- `wavelet`: wavelet's configuration dictionary;
  - `type`: the wavelet's type ('dwt' or 'packet'). **MUST** be written in lowercase. Code was tested only using 'dwt';
  - `name`: the wavelet's name. **MUST** be written exactly as used in the PyWavelets library. E.g.: 'db4';
  - `level`: the wavelet's max level depth. E.g.: 4;
  - `mode`: the wavelet's mode. **MUST** be written exactly as used in the PyWavelets library. E.g.: 'symmetric'.
- `feature`: feature's configuration dictionary;
  - `name`: which feature will be extracted ('mfcc' or 'mel\_spectrogram'). **MUST** be written in lowercase;
  - `n_fft`: the number of n_fft. E.g.: 1024;
  - `hop_length`: the hop length. E.g.: 512;
  - `n_mels`: the number of mels. Only required when using `mel\_spectrogram` as feature extraction. E.g.: 64;
  - `n_mfcc`: the number of mfccs. Only required when using `MFCC` as feature extraction. E.g.: 64;
  - `f_min`: the minimum frequency. E.g.: 0;
  - `f_max`: the maximum frequency. E.g.: 4000;
- `data_augmentation`: data augmentation's configuration dictionary;
  - `mode`: where the feature will be applied on ('feature' or 'raw_audio'). **MUST** be written in lowercase;
  - `target`: which classes the data augmentation will be applied on ('all', 'majority' or 'minority'). The options 'majority' and 'minority' are available only when used on the `CORAA` dataset. **MUST** be written in lowercase; 
  - `p`: the probability of the data augmentation technique(s) being applied. **MUST** be a float number between 0 and 1. E.g.: 1;
  - `techniques`: a dictionary where the keys are the name of the data augmentation technique ('specaugment', 'cutmix', 'specmix', 'audioaugment' or 'denoiser'). The option 'audioaugment' can only be applied when the `mode` is set to `raw_audio`. **MUST** be written in lowercase. Please check the `examples` folder to see which arguments must be passed for each technique.
- `model`: model's configuration dictionary;
  - `name`: which model architecture will be used ('[qiuqiangkong](https://github.com/qiuqiangkong/audioset_tagging_cnn)' or '[aghajani](https://www.ije.ir/article_103377.html)' (only the CNN)). **MUST** be written in lowercase;
  - `use_gpu`: use GPU or not;
  - `output_path`: the folder path where the model's checkpoints will be saved (if the folder doesn't exist, then it will be created);
  - `batch_size`: the batch size. E.g.: 16;
  - `learning_rate`: the learning rate. E.g.: 0.001;
  - `epochs`: the number of epochs. E.g.: 100;
  - `use_lr_scheduler`: use learning rate step scheduler or not (every 10 epochs the learning rate will decay the gamma value of 0.1 unit).
- `kfold`: kfold's configuration dictionary;
  - `num_k`: the number of folders. If zero, then a normal split will be applied. E.g.: 5;

![Mode explanation](/images/modes.png)

### Running the Code

#### Using Virtual Environment

To train the model, run the following command:
```python
python3 train.py -c CONFIG_FILE_PATH
```

To run the model's inference, run the following command (the configuration file **MUST** be the same one used above):
```python
python3 test.py -c CONFIG_FILE_PATH
```

#### Using Docker (Recommended)

To train the model, run the following command:
```bash
docker run ser-wavelet \
       /bin/bash -c "python3 train.py -c CONFIG_FILE_PATH" \
       --gpus all --runtime=nvidia -it
```

To run the model's inference, run the following command (the configuration file **MUST** be the same one used above):
```bash
docker run ser-wavelet \
       /bin/bash -c "python3 test.py -c CONFIG_FILE_PATH" \
       --gpus all --runtime=nvidia -it
```

## Paper's Results

We employed a speaker-dependent approach and cross-validation technique to partition the dataset into five folds. In this setup, 80% of the data constituted the training set, while the remaining 20% formed the validation set. Throughout these folds, we ensured that the class distribution remained consistent with that of the entire dataset, and the same fold structure was applied uniformly across all models to ensure fair result comparisons. It is worth noting that we solely employed the training data for the split, reserving the test set for model evaluation at PROPOR 2022.

The PANNs models underwent training for 100 epochs, while the CNN model was trained for 50 epochs. A learning rate of 0.001 and a batch size of 16 were applied during the training process. The PANNs 10 model achieved the best performance for both the validation and test sets, attaining the highest F1-Score values of approximately 0.667 and 0.566, respectively. Notably, the model's performance appears to improve as the model size and complexity increase.

As shown in the table below, when comparing our results in the CORAA dataset with those of other studies, we observe that we achieved the second-best performance among all submissions for PROPOR 2022. However, if we focus on methods that do not employ open-set strategies (such as transfer learning, fine-tuning, or utilizing additional datasets to train the model), our proposed approach outperforms them, boasting a 3.1\% higher F1-Score.

![Comparing results for the CORAA dataset](/images/table_coraa.png)

When comparing our results in the EmoDB, SAVEE, and RAVDESS datasets with those of other studies, as presented in the following table, our model’s performance falls short of the best results, particularly for the RAVDESS and SAVEE databases. However, several diverging factors among these works make a direct comparison challenging. Furthermore, when comparing our results with related works that utilize the same speakers, data, and classes as our approach, we observe that our performance now approaches or even surpasses the best results (0.845 for EmoDB, 0.645 for RAVDESS, and 0.594 for SAVEE). Nonetheless, we emphasize that precise comparisons are challenging due to the differences in the approaches used, particularly whether they are speaker-dependent or speaker-independent.

![Comparing results for the rest of the datasets](/images/table_other_datasets.png)

## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Contact

Author: Rafael Greca Vieira - [GitHub](github.com/rafaelgreca/) - [LinkedIn](https://www.linkedin.com/in/rafaelgreca/) - rgvieira97@gmail.com