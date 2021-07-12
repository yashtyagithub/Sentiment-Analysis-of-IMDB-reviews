<p align="center">
<img width="600" src="https://th.bing.com/th/id/OIP.w5f2GKn-IkrCKzFPQVa1yQHaEL?pid=ImgDet&rs=1"
 </p>

# Sentiment-Analysis-of-IMDB-reviews
## Overview 
Sentiment analysis is one of the  natural language processing technique used to determine the subjective opinions or feelings collected from various sources about a particular subject. Business ofter use sentiment analysis tools to determine to understand how customer feel about their product. In this projuct machine learning and deep learning models are trained on huge [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  and  then best model is used to create a web appliction using flask to determine the sentiment of text given by the user.
  
## Dataset
The [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) consists of 50,000 movie or tv shows reviews from IMDB userd that are labeled as positive oe negative. The number of positive and negative reviews in the dataset are equal i.e 25000 positive reviews and 25000 negative reviews. 

## Machine Leaning Models
* **Steps before applying models**
  
  * Preprocessing: 
    1. lower the reviews 
    2. remove punctuations from reviews 
    3. remove HTML tags from reviews
    4. remove extra spaces from reviews 
    5. remove numbers from reviews  
    6. remove special character from reviews 
    7. remove stopwords from the reviews
  
  * Tokenization
    1. represent reviews as a collections of words or tokens.
    2. perform stemming on the tokens.
  
  * Transform reviews into Feature vectors using bag of words
  
  * Transform reviews into TF-IDF vectors
 
* **Apply Logistic Regression**
  
    ![logistic regression](https://user-images.githubusercontent.com/50082770/125299878-2baeef00-e347-11eb-84f4-337b3ca7385a.png)
  
* **Apply Stochastic Gradient Descent Classifier**
  
  ![image](https://user-images.githubusercontent.com/50082770/125300580-cdced700-e347-11eb-917a-07d995c62bd5.png)

* **Apply Multinomial Naive Bayes Classifier**
  
  ![mnb](https://user-images.githubusercontent.com/50082770/125300934-1dad9e00-e348-11eb-83d2-63e8d1ed4017.png)

 * All the steps mentioned above performed in this [jupyter notebook](https://github.com/yashtyagithub/Sentiment-Analysis-of-IMDB-reviews/blob/master/different_models.ipynb)

## LSTM Model
 
  
 * Deep learning text classification model  architectures generally consist of the following components connected in sequence
  
  ![deep learning](https://user-images.githubusercontent.com/50082770/125319351-505f9280-e358-11eb-9ccf-c04bbf9b8558.png)
  
 * Preprocessing: 
    1. lower the reviews 
    2. remove punctuations from reviews 
    3. remove HTML tags from reviews
    4. remove extra spaces from reviews 
    5. remove numbers from reviews  
    6. remove special character from reviews 
    7. remove stopwords from the reviews

 * Reviews are first encoded so that each word is represented by a unique integer by using  tensorflow keras tokenizer and then we make all the review vectors equal to length 128 by     adding padding to  review vectors or by truncating the review vectors.
  
 * **First method** : In the tensorflow keras LSTM  model, embedding layer is the first hidden layer and that layer is initialized with random weights and will learn an embedding for    all of the words in the training dataset during training of the model. 
   * The summary of the model is :
    
    ![lstmsummary](https://user-images.githubusercontent.com/50082770/125331571-c6b6c180-e365-11eb-8545-a26e6a8af786.png)

 *  **Second method** : Use the same netword architecture as in first method but use the pre-trained word2vec 300 dimension word embeddings as initial input. So in this method            embedding layer is not trained during the training of the model.
  
    * The summary of the model is :
  
     ![lstmw2v](https://user-images.githubusercontent.com/50082770/125331146-39736d00-e365-11eb-9b7c-5f3a3773763e.png)

 * All the points mentioned above performed in this [jupyter notebook](https://github.com/yashtyagithub/Sentiment-Analysis-of-IMDB-reviews/blob/master/LSTM.ipynb)
  
## BERT Model
  * BERT stands for Bidirectional Encoder Representations from Transformers and it is a state-of-the-art deep learning model used for NLP tasks.
  
  * We use DistilBert in this project. DistillBert is a small, fast, cheap and ligh Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-        uncased , runs 60% faster while preserving over 95% of BERT's performances
  
  * Preprocessing: 
    1. lower the reviews 
    2. remove punctuations from reviews 
    3. remove HTML tags from reviews
    4. remove extra spaces from reviews 
    5. remove numbers from reviews  
    6. remove special character from reviews 
    7. remove stopwords from the reviews 
 
 * Using the Distilbert tokenizer to give the numerical representation to each word such that it can be provided to distilbert model.
  
 * Configure the loaded DistilBert model and train for fine-tunning.
 
 * Model architecture
  
  ![bertimage](https://user-images.githubusercontent.com/50082770/125338168-7e030680-e36d-11eb-975e-ca7e4f44562e.png)

 * Implementation of DistillBert present in this [jupyter notebook](https://github.com/yashtyagithub/Sentiment-Analysis-of-IMDB-reviews/blob/master/Distillbert.ipynb)
  
## Accuracy of different models on test dataset
  
  ![model vs accuracy](https://user-images.githubusercontent.com/50082770/125339815-53b24880-e36f-11eb-89ed-79376f8b2cb8.png)
 
  **DistilBert** have highest accuracy of 0.89 on test dataset. 

  
  
