from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd


def startup():
  df_fr= pd.read_csv('/home/Hooriya/FinalDataSetStopWordRemoval.csv')
  df_fr.dropna(subset=['Comment'], inplace = True)
  df_fr.dropna(subset=['Class'], inplace = True)

    # Fit the CountVectorizer to the training data
  global vectTrain_fr;
  vectTrain_fr = CountVectorizer(input='content',decode_error='ignore',analyzer='word',ngram_range=(1,1)).fit(df_fr['Comment'])

  global X_train_vectorized_fr ;
  X_train_vectorized_fr = vectTrain_fr.transform(df_fr['Comment'])
  global model_fr;
  model_fr= svm.SVC( kernel ='linear')
  model_fr.fit(X_train_vectorized_fr, df_fr['Class'])
  
  x1 = pd.ExcelFile("/home/Hooriya/ArabiziSenti.xlsx")
  df_arabi = x1.parse("Sheet1")
  df_arabi.dropna(subset=['Comment'], inplace = True)
  df_arabi.dropna(subset=['Class'], inplace = True)

  
  x2 = pd.ExcelFile("/home/Hooriya/ArabicSenti.xlsx")
  df_arabic = x2.parse("Sheet1")
  df_arabic.dropna(subset=['Comment'], inplace = True)
  df_arabic.dropna(subset=['Class'], inplace = True)
  
  global vectTrain_arabizi;
  vectTrain_arabizi = CountVectorizer(input='content',decode_error='ignore',analyzer='word',ngram_range=(1,1)).fit(df_arabi['Comment'])

  global X_train_vectorized_arabizi ;
  X_train_vectorized_arabizi = vectTrain_arabizi.transform(df_arabi['Comment'])
  global model_arabizi;
  model_arabizi= svm.SVC( kernel ='linear')
  model_arabizi.fit(X_train_vectorized_arabizi, df_arabi['Class'])
  
  global vectTrain_arabic;
  vectTrain_arabic = CountVectorizer(input='content',decode_error='ignore',analyzer='word',ngram_range=(1,1)).fit(df_arabic['Comment'])

  global X_train_vectorized_arabic ;
  X_train_vectorized_arabic = vectTrain_arabic.transform(df_arabic['Comment'])
  global model_arabic;
  model_arabic= svm.SVC( kernel ='linear')
  model_arabic.fit(X_train_vectorized_arabic, df_arabic['Class'])
  
  


    
  
startup()
        
