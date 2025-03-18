from NLPUtilsBERT.Utils_TextClassification_BERT import TextClassificationModel

# Configuration
MODEL_FOLDER = "MODEL"

# Make predictions
classifier = TextClassificationModel(model_folder=MODEL_FOLDER)
classifier.load_model()

text = "I love playing football."; print(f"\n{text} : {classifier.predict(text)}")
text = "This is my business place."; print(f"\n{text} : {classifier.predict(text)}")
text = "My Chrome browser is giving issues."; print(f"\n{text} : {classifier.predict(text)}")
