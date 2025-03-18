import pandas as pd

from NLPUtilsBERT.Utils_TextClassification_BERT import TextClassificationModel

# Configuration
dataset_path = r"data/bbc-test.csv"
pretrained_model_name = 'bert-base-uncased'  # Options: 'bert-base-uncased', 'distilbert-base-uncased'
batch_size = 16
learning_rate = 1e-7
num_train_epochs = 50
early_stopping_patience = 5
weight_decay = 0.01
test_size = 0.2
val_size = 0.3
resume_from_checkpoints = True
random_state = 73
MODEL_FOLDER = "MODEL"

# Load dataset
df = pd.read_csv(dataset_path)

# Initialize and train the model
text_classifier = TextClassificationModel(pretrained_model_name=pretrained_model_name,
                                          batch_size=batch_size,
                                          learning_rate=learning_rate,
                                          num_train_epochs=num_train_epochs,
                                          weight_decay=weight_decay,
                                          model_folder=MODEL_FOLDER,
                                          early_stopping_patience=early_stopping_patience,
                                          test_size=test_size,
                                          val_size=val_size,
                                          random_state=random_state,
                                          resume_from_checkpoints=resume_from_checkpoints)

ds_train, ds_val, ds_test = text_classifier.create_datasets(df, target_column="category")
text_classifier.train(ds_train, ds_val)

# Evaluate the model
eval_results = text_classifier.evaluate(ds_test)
print('Evaluation results:', eval_results)
