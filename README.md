# CSE343-ML-Project -  Suicide Ideation Prediction from Social Media Conversations
Course Project for CSE-343 (Machine Learning) - Monsoon 2023

## Project Overview
In the face of growing concerns over mental health and the alarming rise in suicide rates, our project aims to detect and address suicide ideation by analyzing social media conversations. Utilizing advanced machine learning techniques, we've developed a robust model capable of identifying individuals at heightened risk based on their online activities. Our solution includes a real-world application through a Reddit Bot, designed to flag posts with potential suicide ideation risks.

## Team Members
- Medha Hira - medha21265@iiitd.ac.in
- Arnav Goel - arnav21519@iiitd.ac.in
- Siddharth Rajput - siddharth21102@iiitd.ac.in

## Introduction
The project addresses the critical need for effective suicide prevention strategies by leveraging social media as a platform for early detection of suicide ideation. With a 36% increase in suicide rates from 2000 to 2021, our predictive model seeks to provide timely intervention, potentially saving lives by identifying at-risk individuals through their digital footprints. Recognizing the pivotal role social media plays in modern communication, our system is designed to detect suicide ideation through analysis of Reddit posts. Our approach utilizes a comprehensive dataset from the r/SuicideWatch subreddit, applying machine learning algorithms to identify early signs of suicidal thoughts.

## Dataset and Preprocessing
We employed the University of Maryland Reddit Suicidality Dataset, conducting rigorous data preprocessing to clean and prepare text data for analysis. Techniques included removal of non-ASCII characters, URLs, usernames, and punctuation, as well as stopwords and lowercasing for standardization.

![NonSuicideWordCloud](https://github.com/arnav10goel/CSE343-ML-Project/assets/97335445/f6647068-9efa-4f92-88ca-73456c275f06)

![SuicideWordCloud](https://github.com/arnav10goel/CSE343-ML-Project/assets/97335445/deea7619-94cc-4dbd-95b3-0105a54b006f)



## Methodology
Our methodology encompasses a diverse range of machine learning models, including Logistic Regression, SVM, Naive Bayes, Decision Trees, and Random Forest, among others. We also explored ensemble methods and neural networks for enhanced predictive performance. Evaluation metrics such as accuracy, precision, and recall were employed to assess model effectiveness.

## Results and Analysis
Our findings indicate that models like LDA, Logistic Regression, and the SVM classifier perform best, with notable improvements using Word2Vec embeddings. Ensemble methods and a Multilayer Perceptron (MLP) classifier also showed promising results, demonstrating the efficacy of our approach in detecting suicide ideation with high accuracy.

Results for Machine Learning Models:
<img width="881" alt="Screenshot 2024-03-16 at 12 49 16 PM" src="https://github.com/arnav10goel/CSE343-ML-Project/assets/97335445/e925649b-abfd-494b-8a01-baacbe6e0e42">

Results for Ensemble Method and a MLP Classifier:
<img width="981" alt="Screenshot 2024-03-16 at 12 49 44 PM" src="https://github.com/arnav10goel/CSE343-ML-Project/assets/97335445/eba9f53c-ff27-4c7e-ae22-de1927038d1a">

## Model Deployment
Reddit Bot Demo: [YouTube Link](https://www.youtube.com/watch?v=7NHZzQAnS1k)

The culmination of our project is the deployment of a Reddit Bot, integrating our most effective machine learning model to actively scan and flag posts for suicide ideation on Reddit. This bot aims to bridge the gap between at-risk individuals and timely mental health support.
