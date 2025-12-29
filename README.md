# Dissertation_Fake_News_Detection

# Fake News Detection Using Machine Learning and Deep Learning
# Project Overview

This project presents a comparative study of machine learning and deep learning models for automated fake news detection. It was developed as part of an MSc dissertation in Artificial Intelligence at Sheffield Hallam University. The study evaluates how effectively different text classification models can distinguish between real and fake news articles using a unified experimental setup.

# Motivation

The rapid spread of misinformation across digital platforms has created serious societal, political, and public health challenges. Fake news is often deliberately written to resemble legitimate journalism, making it difficult to detect using simple rule-based or surface-level techniques. This project is motivated by the need to develop reliable, automated fake news detection systems that can operate at scale while maintaining strong classification performance and computational efficiency.

# Dataset Description

The experiments are conducted using the WELFake (Web-Enhanced Large Fake News) dataset, which is one of the largest and most balanced publicly available fake news datasets. It consists of over 70,000 English-language news articles collected from multiple credible sources such as Kaggle, Reuters, BuzzFeed Political News, and the McIntire dataset. Each article is labelled as either fake or real, enabling supervised learning under realistic conditions.

# Data Preprocessing

Text preprocessing is applied to ensure data quality and consistency across all models. This includes converting text to lowercase, removing URLs, punctuation, numerical characters, and extra whitespace. These steps help reduce noise and improve the effectiveness of both machine learning and deep learning models. The cleaned text is then prepared using different representation techniques depending on the model type.

# Text Representation

Two text representation approaches are used in this project. For the machine learning baseline, TF–IDF (Term Frequency–Inverse Document Frequency) is employed to convert textual data into numerical feature vectors. For deep learning models, text is tokenised and padded to fixed-length sequences, allowing the models to learn semantic and contextual information directly from word order and sequence structure.

# Models Implemented

This project compares one traditional machine learning model with three deep learning architectures. Logistic Regression is used as a supervised machine learning baseline due to its efficiency, interpretability, and strong performance in text classification tasks. The deep learning models include Convolutional Neural Networks (CNN) for capturing local textual patterns, Long Short-Term Memory (LSTM) networks for modelling sequential dependencies, and Bidirectional LSTM (BiLSTM) models for enhanced contextual understanding through forward and backward sequence processing.

# Experimental Setup

All models are trained and evaluated using the same dataset split, preprocessing pipeline, and evaluation metrics to ensure a fair comparison. The dataset is divided into training and testing sets using an 80:20 ratio with stratified sampling to preserve class balance. This controlled experimental design ensures that observed performance differences are due to model architecture rather than data inconsistencies.

# Evaluation Metrics

Model performance is assessed using standard classification metrics, including Accuracy, Precision, Recall, F1-score, and AUC–ROC. These metrics provide a comprehensive view of each model’s effectiveness, particularly in handling class balance and identifying fake news accurately.

# Results and Findings

The results demonstrate that CNN achieves the best overall performance, outperforming other models in terms of accuracy, F1-score, and AUC–ROC. BiLSTM also performs strongly, benefiting from bidirectional contextual learning. Logistic Regression establishes a reliable baseline with competitive results, while LSTM shows unstable performance in this setup. Overall, CNN offers the best balance between classification accuracy and computational efficiency.

# Significance of the Study

This study contributes to fake news detection research by providing a controlled and reproducible comparison of machine learning and deep learning models using a single benchmark dataset. The findings are valuable for both academic researchers and practitioners, particularly those seeking efficient and scalable solutions for misinformation detection without relying on computationally expensive transformer-based models.

# Limitations

The scope of this project is limited to English-language textual data and does not include multimodal information such as images, videos, or social network metadata. Transformer-based models such as BERT are excluded due to computational constraints. Real-time deployment and adversarial robustness are also outside the scope of this study.

# Future Work

Future research can extend this work by incorporating transformer-based architectures, multilingual datasets, adversarial attack resistance, or hybrid models. Further evaluation on real-time data streams and deployment-focused optimisation can also enhance practical applicability.
