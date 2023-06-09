from src.utils import create_propor_train_dataframe, labels_mapping
from src.processing import processing

if __name__ == "__main__":
    train_df = create_propor_train_dataframe()
    train_df = labels_mapping(
        df=train_df,
        dataset="propor2022"
    )
    
    X_train, y_train = processing(
        df=train_df,
        to_mono=True,
        sample_rate=16000,
        apply_one_hot_encoder=True
    )
    
    print(X_train.shape, y_train.shape)