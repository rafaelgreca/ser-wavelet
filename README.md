# Speech Emotion Recognition using Deep Learning and Wavelet Transform

Repository dedicated to the developed solution for the end-of-course work of University of São Paulo's [Master in Business Administration (MBA) in Artificial Intelligence and Big Data](https://mba.iabigdata.icmc.usp.br/) program. The solution main focus is to tackle the Speech Emotion Recognition task and is composed by a Convolutional Neural Network based on [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) trained using Mel Spectrogram and Discrete Wavelet Transform.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install this package, firstly clone the repository to the directory of your choice using the following command:
```bash
git clone https://github.com/rafaelgreca/ser-wavelet.git
```

Finally, you need to create a conda environment and install the requirements. This can be done using conda or pip. For `conda` use the following command:
```bash
conda create --name ser --file requirements/conda.txt python=3.8.10
conda activate ser
```

For `pip` use the following command:
```bash
conda create --name ser python=3.8.10
conda activate ser
pip install -r requirements/pip.txt
```

## Getting Started

After installing the dependencies you are ready to run the code. But first, lets understand the code and its inputs arguments, shall we?

### Directory Structure

```bash
./
├── config/
│   ├── mode_1.json
│   └── mode_2.json
├── __init__.py
├── LICENSE
├── notebooks/
│   ├── analysis.ipynb
│   ├── eda.ipynb
│   ├── parsifal-analysis.ipynb
│   └── visualize-training.ipynb
├── README.md
├── requirements/
│   ├── conda.txt
│   └── pip.txt
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
├── test.py
└── train.py
```

Explaining briefly the main folders and files:

- `config`: where the model and feature extraction configurations files (**MUST** be in a JSON format) that will be used as inputs arguments are saved;
- `notebooks`: where the notebooks used to visualize the training/validation step, do the exploratory data analysis (EDA) and to create the analysis used for the thesis writing are saved;
- `requirements`: where the requirements files for the installation step are saved.
- `src`: where the core functions are implemented, such as: the data augmentation, feature extraction, audios preprocessing steps, and the models/datasets creation;
- `test.py`: the main file responsible to inference the model on new data;
- `train.py`: the main file responsible to train the model.

### Inputs Parameters

The following arguments **MUST** be inside the configuration JSON file (exactly as structured in the files inside the `config` folder).

- `input_path`: the dataset's folder path (**MUST** exists);
- `output_path`: the folder where the extracted features will be saved (if the folder doesn't exist, then it will be created);
- `sample_rate`: the audio's sampling rate. E.g.: 8000, 16000, 24000 and so on;
- `dataset`: which dataset will be used ('coraa', 'savee', 'emodb' or 'ravdess'). **MUST** be written in lowercase;
- `mode`: which mode will be run ('mode\_1' or 'mode\_2'). **MUST** be written in lowercase;
- `overwrite`: overwrite the extracted features or not;
- `to_mono`: convert the audios to mono or not;
- `wavelet`: wavelet's configuration dictionary;
- `feature`: feature's configuration dictionary;
- `data_augmentation`: data augmentation's configuration dictionary;
- `model`: model's configuration dictionary;
- `kfold`: kfold's configuration dictionary;

### Running the Code

To train the model, run the following command:
```python3
python3 train.py -c CONFIG_FILE
```

To run the model's inference, run the following command (the configuration file **MUST** be the same one used above):
```python3
python3 test.py -c CONFIG_FILE
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Contact

Author: Rafael Greca Vieira - [GitHub](github.com/rafaelgreca/) - [LinkedIn](https://www.linkedin.com/in/rafaelgreca/) - rgvieira97@gmail.com