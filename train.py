import os
from src.dataset import create_dataloader
from src.utils import create_propor_train_dataframe, prepare_propor_test_dataframe, labels_mapping, save, read_feature
from src.processing import processing, split_data

sample_rate = 16000
k_fold = 5
to_mono = True
dataset = "propor2022"
feat_output = "./features"
max_samples = 15 * sample_rate
feat_config = {
    "sample_rate": sample_rate,
    "feature": "mfcc",
    "n_fft": 1024,
    "hop_length": 512,
    "n_mfcc": 128
}
wavelet_config = {
    "name": "db4",
    "type": "dwt",
    "level": 4,
    "mode": "symmetric"
}

def extract_features():
    # reading the training dataset
    train_df = create_propor_train_dataframe()
    train_df = labels_mapping(
        df=train_df,
        dataset=dataset
    )
    
    # preprocessing the training dataset
    X_train, y_train = processing(
        df=train_df,
        to_mono=to_mono,
        sample_rate=sample_rate,
        max_samples=max_samples
    )
    
    # splitting the training dataset into training and validation (and saving)
    split_data(
        X=X_train,
        y=y_train,
        dataset=dataset,
        output_path=feat_output,
        k_fold=k_fold
    )
    
    # reading the test dataset
    test_df = prepare_propor_test_dataframe()
    test_df = labels_mapping(
        df=test_df,
        dataset=dataset
    )
        
    # preprocessing the test dataset
    X_test, y_test = processing(
        df=test_df,
        to_mono=to_mono,
        sample_rate=sample_rate,
        max_samples=max_samples
    )
    
    # saving the test features
    folder_path = os.path.join(feat_output, dataset)
        
    save(path=folder_path, name="X_test", tensor=X_test)
    save(path=folder_path, name="y_test", tensor=y_test)

if __name__ == "__main__":
    extract_features()
    
    if k_fold > 0:
        for fold in range(k_fold):
            # reading training audio features
            X_train = read_feature(
                path=os.path.join(feat_output, dataset),
                fold=fold,
                name="X_train.pth",
            )
            
            y_train = read_feature(
                path=os.path.join(feat_output, dataset),
                fold=fold,
                name="y_train.pth",
            )
            
            training_dataloader = create_dataloader(
                X=X_train,
                y=y_train,
                feature_config=feat_config,
                wavelet_config=wavelet_config,
                batch_size=32,
                num_workers=0,
                shuffle=True,
                training=True
            )
            
    else:
        # reading training audio features
        X_train = read_feature(
            path=os.path.join(feat_output, dataset),
            name="X_train.pth",
        )
        
        y_train = read_feature(
            path=os.path.join(feat_output, dataset),
            name="y_train.pth",
        )