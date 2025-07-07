from bertopic import BERTopic

model = BERTOpic.load('Model_training/BERTopic_model.pkl')
model.save("Model_training/BERTopic_model_persisted", serialization = 'persistence')
print("Model re-saved")
