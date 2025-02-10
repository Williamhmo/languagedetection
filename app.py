import streamlit as st
import tensorflow as tf
import numpy as np
import keras
from keras.layers import TextVectorization
from keras.utils import pad_sequences
import pickle
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components

def detect(userinput):
  language_result=[]
  inputsentences=[]
  inputsentence=''
  for i in userinput:
    if i !='.' and i!='။' and i!='。':
      inputsentence+=i
    else:
      inputsentences.append(inputsentence)
      inputsentence=''
  model=keras.models.load_model("LanguageDetectionCNNmodel2.h5")
  sentences=[]
  sentence='' 
  for i in userinput:
    if i !='.' and i!='။' and i!='。':
      sentence+=' '+i
    else:
      sentences.append(sentence)
      sentence=''
      
  if st.button("Detect"):
    if userinput is None:
        st.warning("Please enter sentence(s) before classifying!")
    else:
      tokenizer=pickle.load(open('pickled_tokenizer2.pkl','rb'))
      tokenizer.fit_on_texts(sentences)
  # word_index = tokenizer.word_index
  # st.write(word_index)
      testing_sequences = tokenizer.texts_to_sequences(sentences)
      testing_padded = pad_sequences(testing_sequences,maxlen=100,padding='post')
      result0=model.predict(testing_padded)
      result=result0.argmax(axis=1)
      classes=['Burmese','English','Japanese','Shan','Chinese']
      for i in result:
          language_result.append(classes[i])
      final=pd.DataFrame({'Text':inputsentences,'Language':language_result})
      final
      
def background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://www.shutterstock.com/shutterstock/videos/1094703853/thumb/7.jpg?ip=x480.png");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
  
def contact():
    contact_form="""<form action="https://formsubmit.co/hlaingminoo29917@gmail.com" method="POST">
    <input type="text" name="name" placeholder="Name "required>
    <input type="email" name="email" placeholder="Enter email address">
    <textarea id="subject" name="subject" placeholder="Your message here..." style="height:200px"></textarea>
    <input type="hidden" name="_captcha" value="false">
    <button type="submit">Send</button>
    </form>
    <style>
input[type=text],input[type=email], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
  margin-top: 6px;
  margin-bottom: 16px;
  resize: vertical;
}
button[type=submit] 
{
  background-color: #D1E5F3;
  color: black;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
button[type=submit]:hover
{
  background-color: #2E34DA;
  color = white;
}
</style>    
    
    """
    st.markdown(contact_form,unsafe_allow_html=True)


def main():
    menu=['Home','Detect Language','Contact developer']
    sidebarImg=Image.open('pic1.jpg')
    st.sidebar.image(sidebarImg)
    choice=st.sidebar.selectbox('Menu',menu)
    
    if choice == 'Home':
        st.header('Welcome to our Language Detection App!')
        homeImg=Image.open('Language Classify cover photo.jpg')
        # st.image(homeImg)
        background()
        # st.write('Language is so important to commmunicate one another.')
        
    elif choice == 'Detect Language':
      st.header('Welcome to our Language Detection App!')
      # st.subheader('LANGUAGE DETECTION '
      # sidebarImg=Image.open(r'D:\ML Project\pic\pic3.jpg')
      # st.image(sidebarImg)
      st.write('Only 5 languages (Burmese, English, Japanese, Shan, Chinese) are available now ...')
      st.write('Note : Use one of these [full stop(.) , ပုဒ်မ (။) , period( 。)] to end the sentence..:')
      
      try:
        userinput=st.text_input('Please enter sentence(s)','မင်္ဂလာပါ ။ Welcome.')
        if userinput[-1]!='.' and '။' and '。':
          userinput+='.'
        else:
          userinput=userinput

        
        inputsentences=[]
        inputsentence=''
        for i in userinput:
          if i !='.' and i!='။' and i!='。':
            inputsentence+=i
          else:
            inputsentences.append(inputsentence)
            inputsentence=''
        option = st.sidebar.selectbox("Classifier Options ",('Logistic Regression Classifier','Dicision Tree Classifier','Random Forest Classifier','XGBoost Classifier','Naive Bayes Classifier','Convolutional Neutral Network'))
        
        if option=='Naive Bayes Classifier':
          sentences=[]
          sentence=''
          for i in userinput:
            if i !='.' and i!='။' and i!='。':
              sentence+=i
            else:
              sentences.append(sentence)
              sentence=''
          pickled_NB = pickle.load(open('model_NB.pkl', 'rb'))
          if st.button("Detect"):
            if userinput is None:
                st.warning("Please enter sentence(s) before classifying!")
            else:
                result=pickled_NB.predict(sentences)
                final_result=pd.DataFrame({'Text':inputsentences,'Language':result})
                final_result
      
        elif option=='XGBoost Classifier':
          sentences=[]
          sentence=''
          for i in userinput:
            if i !='.' and i!='။' and i!='。':
              sentence+=i
            else:
              sentences.append(sentence)
              sentence=''
          pickled_xgb=pickle.load(open('model_xgb.pkl', 'rb'))
          if st.button("Detect"):
            if userinput is None:
                st.warning("Please enter sentence(s) before classifying!")
            else:
                result=pickled_xgb.predict(sentences)
                final_result=pd.DataFrame({'Text':inputsentences,'Language':result})
                final_result
  
        elif option=='Random Forest Classifier':
          sentences=[]
          sentence=''
          for i in userinput:
            if i !='.' and i!='။' and i!='。':
              sentence+=i
            else:
              sentences.append(sentence)
              sentence=''
          pickled_RF = pickle.load(open('model_RF.pkl', 'rb'))
          if st.button("Detect"):
            if userinput is None:
                st.warning("Please enter sentence(s) before classifying!")
            else:
                result=pickled_RF.predict(sentences)
                final_result=pd.DataFrame({'Text':inputsentences,'Language':result})
                final_result
        
        elif option=='Dicision Tree Classifier': 
          sentences=[]
          sentence=''
          for i in userinput:
            if i !='.' and i!='။' and i!='。':
              sentence+=i
            else:
              sentences.append(sentence)
              sentence=''      
          pickled_DT = pickle.load(open('model_DT.pkl', 'rb'))
          if st.button("Detect"):
            if userinput is None:
                st.warning("Please enter sentence(s) before classifying!")
            else:
                result=pickled_DT.predict(sentences)
                final_result=pd.DataFrame({'Text':inputsentences,'Language':result})
                final_result

        elif option=='Logistic Regression Classifier':  
          sentences=[]
          sentence=''
          for i in userinput:
            if i !='.' and i!='။' and i!='。':
              sentence+=i
            else:
              sentences.append(sentence)
              sentence=''     
          pickled_LR = pickle.load(open('model.pkl', 'rb'))
          if st.button("Detect"):
            if userinput is None:
                st.warning("Please enter sentence(s) before classifying!")
            else:
                result=pickled_LR.predict(sentences)
                final_result=pd.DataFrame({'Text':inputsentences,'Language':result})
                final_result
        
        elif option=='Convolutional Neutral Network':
          detect(userinput)
      except IndexError:
        st.write("There is no text !" )
        
    elif choice=='Contact developer':
        contact()           
    
    
    
    
if __name__ =='__main__':
    main()
