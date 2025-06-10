The repository must be explored as a workflow.

Phase 1 of the project, which took place in the SP2025 semester is found in the Sentiment_Analysis.py code. This code is uploaded to the Streamlit cloud where it is still operable and running. 
The completed Phase 2 of the project, which is taking place in the SU2025 semester will be uploaded to the Streamlit cloud along with the previous code.

Stages 1 and 2 of the summer project are found inside the Online_Extraction directory. Here is a quick summary of the codespaces found:
  1. News.py. This is the code that houses the functions that will continually run at midnight to extract news articles related to Tulane University via a connection with 
      News API and setiment analysis by Gemini API.
  2. Tweets.py. This is the code that houses the functions that will continually run at midnight to extract X posts related to Tulane University via a connection with 
      X API and setiment analysis by Gemini API.
  3. rss.py. This is the code that houses the functions that will run to extract the RSS Feeds from several news sources, filters them by keywords, and extracts entities via entity
      recognition using SpaCy. This code cannot be run continually or on GithubActions and can only be run on a local machine given the use of Selenium being limited on Github Actions.
  4. run.py. This is the code that combines and actually runs the functions defined on the previous codes. This is the code that is being run every day at midnight.

Stage 3 of the summer project is found inside the Model_training directory.
  1. matrix.py. Houses the first iteration, untrained sentence transformer machine learning model.
  2. model_train.py. Houses the second iteration, first trained model. It was trained on the articles_
