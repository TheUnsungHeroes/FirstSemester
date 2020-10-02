# Class 5: Scikit-learn and Vectorization


### TL:DR
 - Required:
   - If you found the python `class` problematic spend 30 minutes watching [this](https://www.youtube.com/watch?v=ZDa-Z5JzLYM) and [this](https://www.youtube.com/watch?v=BJ-VvGyQxho). It is well worth your time. Please code along while you watch it.
   - Have your performance metrics from the assignment ready for class
 - Highly recommended:
   - In your study group split up and watch one video each on a machine learning method and meet back up and briefly present the model to each other. See the link in the subsection.
   - Watch either a [the short video](https://www.youtube.com/watch?v=rvVkVsG49u) or [article](https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/) on scikit-learn.
 - Optional
   - Watch this [introduction to scikit-learn](https://www.youtube.com/watch?v=pqNCD_5r0IU&t=8s) (3 Hours)



---

## Plan for Class
- Look at performances of the assignment, who performed best and why? 
- Example of Naive Bayes classification of the IMDB dataset using scikit-learn
  - Vectorization (this is next week topic, but we will thouch upon it briefly)
- Finding the best model

---

## Understanding Python Classes
Some said that they had trouble understanding python classes. If you are one of those I recommend watching [this series](https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc) on Object Oriented Programming (OOB) i python. At least the first to lessons are well worth your time.

## Scikit-learn
In this class I will introduce scikit-learn, a machine learning library for Python if you have the time I recommend watching [this <5 min video](https://www.youtube.com/watch?v=rvVkVsG49uU) or read [this short article](https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/) as an introduction. A much more extensive introduction is this [3 hour introduction](https://www.youtube.com/watch?v=pqNCD_5r0IU&t=8s). It is a bit extensive, but if are completely unfamiliar with scikit-learn this will bring you up to speed in what I consider a relatively short time. 

## Machine Learning methods
Scikit-learn contains a variety of Machine learning methods and while we in class will introduce use them with only minor introduction it is a good idea to have some idea what these model do. Therefore I suggest that in your studygroup you each watch a video on a machine learning method and present them to eachother. Here is a list of recommendation

* [Support Vector Machines (SVM)](https://www.youtube.com/watch?v=efR1C6CvhmE). These fit into a larger group called kernel models and was the big machine learning thing before Neural networks had their breakthrough in 2012.
* [Neural Network (NN)](https://www.youtube.com/watch?v=CqOfi41LfDw). As you probably know this is THE big thing currently and this video by know mean show the full complexity of it. If you are really interested in learning more about this I can suggest [this free online book](http://neuralnetworksanddeeplearning.com) by Michael Nielsen and if you want a more strict introduction you can also look into [the free deep learning book](https://www.deeplearningbook.org) from MIT press.
* [Adaboost](https://www.youtube.com/watch?v=LsK-xG1cLYA). This was the initial introduction to a group of machine learning models called gradient boosting methods, which are still used today and typically perform very well with only little fine-tuning. The athors also won a Gödel prize for their work on this.
* [Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk). A incredible simple model with powerful use cases.
* [Naive Bayes (NB)](https://www.youtube.com/watch?v=O2L2Uv9pdDA). You know this, it is simply included here for completeness. Naive Bayes is the simplest classification solution derived from probability theory.
* [Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog). An extension to NB
* [Logistic regression](https://www.youtube.com/watch?v=yIYKR4sgzI8). If you have used R and LMER for classification you have used logistic regression. It is very popular and is an adaption of linear regression to classification problems.
* [K-nearest Neigbors (KNN)](https://www.youtube.com/watch?v=HVXime0nQeI). Quite a different approach to the other methods as this method does not learn, but infer from tagged data. 

Note: Each of these method could affort their own lecture so don't see these video as the full picture, but they should give a good understanding of the general idea.

## Using your own data
In this class we will be doing classification using Naive Bayes. You have any kind of text data that you would like ot classify using this method feel free to bring that instead.

---

## Materials used in Class
[classification.py](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_05/classification.py)

[classification_additional.ipynb](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_05/classification_additional.ipynb)

<!--
* imdb help
* scikit-learn intro (using diff classifiers, setting hyperparameters)
* Hyperparameter search
* making BOW-using scikit-learn
-->