from bertopic import BERTopic

model = BERTopic.load('Model_training/BERTopic_model.pkl')
model.save("Model_training/BERTopic_model_persisted", serialization = 'persistence')
print("Model re-saved")
