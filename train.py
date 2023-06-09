import os
from src.utils import create_propor_train_dataframe, prepare_propor_test_dataframe, labels_mapping, save
from src.processing import processing, split_data

if __name__ == "__main__":
    sample_rate = 16000
    k_fold = 5
    to_mono = True
    dataset = "propor2022"
    feat_output = "./features"
    max_samples = 15 * sample_rate
    
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
        
    save(path=folder_path, name="X_test", tensor=X_train)
    save(path=folder_path, name="y_test", tensor=y_train)