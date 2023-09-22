# Billboard Rank Prediction
Predict the rank a song will reach on Billboard Hot 100 

## Process
Webscrapes top 100 billboard list as well as corresponding youtube song audio files. 
Featurizes song lyrics (scraped from Genius.com and cleaned) using Word2Vec Google News 300 api through gensim. 
Featurizes the audio file utilizing librosa to compute the following time-varying audio features:zcr, centroid, bandwidth, contrast, rolloff, flatness, flux, mfccs, chroma.
Train a sequence classification with LSTM RNN utilizing TensorFlow.




