# multimodal_churn_model
Our study, for the first time, aims to predict customer churn propensity by uniquely incorporating diverse modalities. This includes analysis of survey data on customer financial literacy level, emotion detection data in the voice of the customer (VOC), and customer relationship management (CRM) data to represent churn risk better.

The proposed multimodal churn model leveraged customer segmentation by detecting negative emotions in CVs during interactions with call center operators, subsequently tailoring the response to address specific customer concerns and requirements, and gaining insights into potential dissatisfaction or frustration causes. 

In the FL model, we utilized a novel SMOGN-COREG supervised model to infer customers' financial literacy levels from their financial behavioral data. The baseline churn model, developed on a robust combination of SMOTE and ensemble ANN algorithms, accurately predicted churn propensity in the context of massive and high-dimensional data. The SER model tackled the power of a pre-trained CNN-VGG16 to discern customers' emotions from their voice attributes, providing yet another layer of understanding of customer behavior.
The results revealed a significant correlation between negative emotions, low Financial Literacy (FL) scores, and an increased risk of churn. We evaluated various fusion learning techniques, including hybrid and late fusion, as well as a non-fusion baseline approach. The hybrid fusion method was found to be the most effective, yielding a Mean Average Precision (MAP) and Macro-Averaged F1 Score of 66% and 54%, respectively.


The detailed Python code for our proposed framework with further result visualization will be uploaded after the paper is published at the conference soon. 

Please cite the paper (below DOI) if you are interested in the proposed methodological approach

DOI:
