# The project
Before proceeding on the how I've successfully implemented this project on this ReadMe, I'd like to shed a little light on what fake news is and what it entails.

According to a recent article on ijert.org, In our modern eran where the internet is ubiquitous, everyone relies on various resources for news, Along with the increase in the use of social media platforms like Facebook, Twitter etc, news spread rapidly among millions of users within a very short span of time. Fake news actually is a type of yellow journalism according to Data-Flair,  Fake news encapsulates pieces of news that may be hoaxes and is generally spread through social media and other online media.

This project basically aims to perform fake news detection via machine learning binary classification with the classification algorithms on data which was obtained from UCI website.

# Methodology
The data obtained for the project in two csv files, one files contains all the categories that are labelled True and another, Fake. However I merged these two csv files with df.append function to give a single dataset and shuffled at the same time.

The features contained in this data include
'title' - The news article title or title of each article sample
'text' - content of the news article
'date' - Date article was probably released
'subject' - subject of the news article, what each article talks about
and finally the 'category' - literally the class, the dependent variable

However, all these features are not required in bulding th detection model and there are reasons why,
The Date feature is irrelevant to the class, whether or not a news article is fake or real has nothin to do with the date it was released, eitherway this feature is just a means to an end
The subject feature has zero or near zero variance and the category feature depends too strongly in it like its the only available feature
The title feature wont matter too much to whether or not the article is fake. In essence it could also be one of the required features but issues with tranforming the data with it and most relevant column; 'text', the article itself is complicated, and finally we are left the 'text' feature and the 'category' as the dependent variable

Next is to transform the data via count vectorizer and tfidf transformer, splitting the data with the model selection train test split and passing in the train data through some classification learning algorithms, In this case
- Logistic Regression
- Ensemble XGBoost
- Passive Agressive Classifier
- And the Bayesian algorithms (Multinomial and Gaussian)

XGBoost and Passive seemed to produce the best accuracy results and evaluation metrics and are both pickled. Reason is to finally conclude with which ever model that may seem to work fine in the automation with test data.

I deployed the model using the streamlit framework as in the mode_deployment python script among the file in this reposistory.

Feel free to email me at barrychukwu12@gmail.com if you find a hoaxed in my notebook or scripts. Thanks for reading
