# Depression or Shower Thoughts: that is the question

# <font color='red'>WARNING</font>

**Content Disclaimer**: This dataset contains real title posts scraped from the r/depression subreddit, and some of the titles contain language that is not safe for work, crude, or offensive. The full dataset is available as `depression.csv`, `preprocessed_thoughts.csv`, `thoughts.csv`, and `token_df.csv`. Unfortunately, I did not provide a sanitized version of my dataset because the words contained were important for the analysis and understanding of the model. Please note that the model, the dataset, and the techniques used are not perfect. If you have any concerns about working with this dataset, looking at my analysis, or the topic in general, you can skip my content overall or click [here](http://iamsorry.com/).


### __Table of Contents__ 
**ignore hyperlinks, changed due to transfer**

- [1-DataWrangle-Preprocessing](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/1-DataWrangle-Preprocessing.ipynb)
  - Contains code for extracting the post titles and the preprocessing they underwent before EDA
- [2-EDA](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/2-EDA.ipynb)
  - EDA on the title of posts using `CountVectorizer()` as my tokenizer
- [2.1-EDA-OwnTokens](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/2.1-EDA-OwnTokens.ipynb)
  - EDA on my lemmatized/porterstemmed tokens
- [3-Modeling](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/3-Modeling.ipynb)
  - Has multiple pipelines, gridsearches and models that include RandomForests, LogisticRegression, and MultinomialNB on the train and test datasets
- [4-Model-Comparisons](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/4-Model-Comparisons.ipynb)
  - Compares what I considered my best model, the random forest model, and a wildcard model that I thought had good results with some exceptions
- [Python File](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/p3_tools.py)
  - This file contains functions I created to kind of automate and pre-clean subreddit posts before it went through my pre-processing and cleaning
- [Datasets](https://git.generalassemb.ly/laternader/project_3/tree/master/deliverables/saved_data)
  - A folder containing all the data that was extracted from Reddit and used on my models 
- [Pickled Assets](https://git.generalassemb.ly/laternader/project_3/tree/master/deliverables/assets)
  - Pickled models and results to help with model comparisons.
  
---  
## Problem Statement

According to the *National Institute of Mental Health*, depression (or clinical depression) is a "mood disorder that causes distressing symptoms that affect how you feel, think, and handle daily activities, such as sleeping, eating, or working" [[source](https://www.nimh.nih.gov/health/publications/depression/index.shtml)]. To be diagnosed with depression, these feelings and symptoms must be prevalent for 2 weeks consistently. We want to be able to distinguish this as early as possible since people with depression have difficulty dealing with depression or being unsure on how to reach out [[source](https://www.healthline.com/health/how-to-help-a-depressed-friend#take-care-of-yourself)]. 
Some signs of depression include:
- Persistent sad, anxious, or “empty” mood
- Feelings of hopelessness or pessimism
- Feelings of guilt, worthlessness, or helplessness
- Loss of interest or pleasure in hobbies or activities
- Decreased energy, fatigue, or being “slowed down”
- Difficulty concentrating, remembering, or making decisions
- Difficulty sleeping, early-morning awakening, or oversleeping
- Appetite and/or weight changes
- Thoughts of death or suicide or suicide attempts
- Restlessness or irritability
- Aches or pains, headaches, cramps, or digestive problems without a clear physical cause and/or that do not ease even with treatment

The techniques of detecting depression vary over a number different ways. One probable way of detecting the signs of depression is through writing. 

My task is to use my free access and knowledge of classification models to distinguish posts from Reddit in whether they are from `r/depression` and `r/Showerthoughts`. The posts from the depression subreddit range from a person's day, to a person asking for help; the intensity of these posts tend to match the range of randomness that shower thoughts present in that subreddit. The selection of shower thoughts was selected as a comparable subreddit was to factor in the randomness of a thought the average person can have. The shower thoughts subreddit is filled with many random thoughts that either could have come from sheer curiosity, celebrations during April 20th or possibly other feelings and emotions built up inside. Whatever the case is, they are definitely random. Here is an example post:
```python
Negative numbers are actually bigger than positive numbers because they have a minus sign at the front.
```

---
## Data Collection
The data comes from one of the biggest online forums, Reddit. I was tasked with webscraping from two subreddits of my choice. The subreddits I selected were **r/depression** and **r/Showerthoughts**. I collected 10,000 posts from each subreddit throught the API, PushShift.

## Preprocessing
I created a function that would automatically scrape all 20,000 posts from the subreddits and return the subreddit the post was from, and the post title and subtitle. From this point, I concatenated the two dataframes `shower_thoughts` and `depression` and called it `thoughts`. 

Within `thoughts`, I remapped the *subreddit* column, where if it came from **r/depression** it was **1** and if it was from **r/Showerthoughts** it was **2**.

After remapping, I took the lengths of each *title* and created *word_count* and *char_count*. 

I also dropped the *selftext* column because there was a lot of deleted/removed posts. Another thing to note was that **r/depression** most of the time had some text in subtitle while **r/Showerthoughts** rarely or had none.

In the middle of this process, I created a couple columns(this is noted as "Token" Processing since I had no idea if this was asked for):

- *tokens*: column contains a list for every title split up using tokenization through regular expressions
- *lem_tokens*: contains a list of the words that were <font color = 'green'>lemmatized</font> through `WordNetLemmatizer`
- *pstem_words*: contains a list of the words that were <font color = 'green'>stemmed</font> through `PorterStemmer`
- *stopwords*: contains sets of words that were found in the post title that was also a stop word from `nltk`

At the end of the preprocessing, I ended up with 5 different csv files on which to create models on:
- `depression.csv`: contains 10,000 posts from the **r/depression** subreddit
- `shower_thoughts.csv`: contains 10,000 posts from the **r/Showerthoughts** subreddit
- `thoughts.csv`: contains the concatenated version of the of both `depression` and `shower_thoughts`
- `preprocessed_thoughts.csv`: contains the same data as `thoughts` but was cleaned; added two columns to represent the word and character count of title and the *selftext* column was dropped
- `token_df.csv`: contains the same data as `preprocessed_thoughts` but has additional columns that display the lemmatized tokens, the PorterStemmed tokens, and the stop words (if any) within in the title

## Data Dictionary
| *column_name* | description | data type | 
| --- | --- | --- |
| *subreddit* | The classification we are predicting through many types of classification models. ***Premap***: contained either `depression` or `Showerthoughts`. ***After-mapping***: contains 1 or 0 representing the subreddits, respectively | ***Premap***: object ***After-mapping***: int |
| *selftext* | The subtitles/descriptions of every subreddit post | object |
| *title* | The main title of a subreddit post. Most of our modelling will be based on fitting to the words contained in a title and determing if it is a shower thought or depression. | object |
| *word_count* | The number of words, or length of title based on number of words. | int |
| *char_count* | The number of characters, including spaces and punctuation, in the title of a post. | int |

---
## Discussion
After completing the preprocessing of the data, I proceeded to look at the descriptive stats. One thing to note was that the maximum amount of characters allowed was 300, with the exception of one post with 301. It seems that the title was limited to users so that the bulk of their post discussion and content could go into self text. This was very prevalent in the `r/depression` subreddit considering most people have a lot to say when it comes to expressing their emotions or displaying ample amounts of sadness, irritation, or frustration through text. This is one reason I dropped the *selftext* column in the models because it would have made the analysis to obvious. I wanted this model to somewhat be applicable (if possible) to other forms of social media, such as Twitter. I know in Twitter, the maximum amount of character input is 140. 

![Distribution of Word Counts](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/imgs/word_dist.png)
*fig. 1*

Another thing I noticed in the descriptive statistics was that there was a lot of repeated post titles in the `r/depression` subreddit. This brought the unique amount of post titles down to __9593__, about 400+ less than the original 10,000 posts scraped. I debated about whether or not I should drop down the duplicates for my models but I assumed the reason for the repetition was that when it came to describing a title in that subreddit, people kept it short because the contents in *selftext* was the main purpose of the post. I kept them to maintain their weight of their words (also because the models I selected for comparisons utilized PorterStemmed words rather than relying on CountVectorizer and TfidVectorizers defaults). 

*fig. 2*
|            Top Titles from "r/depression" | count |
|------------------------:|-------|
|                    Help |    16 |
|                       . |    13 |
|                   Alone |    10 |
|                   Tired |     8 |
|            I’m so tired |     7 |
|              Struggling |     7 |
| I don’t know what to do |     7 |
|              Depression |     7 |
|                      :( |     6 |
|             I need help |     6 |


The distribution graph below adds to the idea that most `r/depression` redditors would put most of their content on *selftext* section of the post. Just looking at the fig. 1 and fig. 2 confirms my assumption on the trend that repeats in `r/depression`. 

I had an assumption with repeating title names, then repeating words must be common. I wanted to figure out what other common words appeared from both subreddits. The outlier being many variants of the word "depress", I wanted to see if there were any trigger words that would also come out of this. I looked at the top 25 PorterStemmed words to figure this out. As I expected, "depress" is the most common word in `r/depression` but it was interesting to see that "help" wasn't as high as I expected considering it is the most common title in the subreddit (fig. 3). This is also might be due to the fact that people with depression tend to not reach out for help as much as we assume they would. The `r/showerthoughts` has "people" as it's top word, the only thing I can assume that the titles there are used to get the attention of people by making a post that could be applicable to everyone.

![Top Words from Depression](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/imgs/depression-words.png)
*fig. 3*

![Top Words from Shower Thoughts](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/imgs/shower-words.png)
*fig. 4*

![Top Words Overall](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/imgs/top-words.png)
*fig. 5*

Once I looked as much as I could for anymore trends, I proceeded to modeling my data. I went through many different hyper parameters and used many models. I used LogisticRegression, MultinomialNB, and RandomForestClassifier as my main models for this project. I also used CountVectorizer and TfidVectorizer to turn my data into a bunch of columns containing 1s and 0s. This was so that I could have a dataframe that was prepped for the classification models.

## Selecting the Best Model
I determined my best model overall was based on these hyperparameters and estimators:
```python
{'model__C': 1,
 'model__penalty': 'l2',
 'trans__max_df': 0.85,
 'trans__max_features': 3000,
 'trans__min_df': 5,
 'trans__ngram_range': (1, 3),
 'trans__stop_words': None}

 Pipeline(steps=[('trans',
                 TfidfVectorizer(max_df=0.85, max_features=3000, min_df=5,
                                 ngram_range=(1, 3))),
                ('model', LogisticRegression(C=1))])
```
One of the reasons I selected this model was because of it's applicability to many other datasets. These were the train and test scores from this model:
```python
Best Train Score: 0.9373880597014925
Best Test Score: 0.9216666666666666
```
As you can see, the variance was not high and neither was the bias. It may not have been accurate as another model I had in mind but the overfitting here was much less, unlike my forest model that I compared it to.

The last main reason why I selected my "best" model was due to the consistency in specificity and sensitivity. I wanted to select a model that remained consistent but was as accurate as possible. I wanted to minimize false positives and false negatives to the best of my ability so I optimized for the best one. The numbers and confusion matrix (fig. 6) below were from my best model:
```python
Best Specificity: 0.9232643118148599 
Best Sensitivity: 0.9200844390832328
```
![Confusion Matrix of the best model](https://git.generalassemb.ly/laternader/project_3/blob/master/deliverables/imgs/bestconf.png)
*fig. 6*


# Why?
These were the top 25 words in determining if a post was from the `r/depression` subreddit as well as their odds and log odds. Just looking at these words we can get an idea about the things people post about online and see if that person is depressed or not. 

*fig. 7*
| word(s) | coefficient | exp_coef |
| --- | --- | --- |
| depress | 9.040511 | 8438.084964 |
| my | 5.688897 | 295.567368 |
| me | 5.258693 | 192.229999 |
| feel | 5.007686 | 149.558261 |
| suicid | 3.840616 | 46.554127 |
| help | 3.816397 | 45.440198 |
| myself | 3.583370 | 35.994655 |
| anyon | 2.918018 | 18.504566 |
| is it | 2.739656 | 15.481652 |
| feel like | 2.697899 | 14.848507 |
| want to | 2.660107 | 14.297823 |
| wish | 2.628148 | 13.848098 |
| friend | 2.547431 | 12.774250 |
| tire | 2.481849 | 11.963363 |
| mental | 2.472815 | 11.855772 |
| im | 2.458320 | 11.685165 |
| sad | 2.432114 | 11.382917 |
| therapi | 2.426240 | 11.316249 |
| happi | 2.308422 | 10.058536 |
| advic | 2.259658 | 9.579815 |
| struggl | 2.173840 | 8.791984 |
| ~~fuck~~ | 2.138048 | 8.482860 |
| vent | 2.137042 | 8.474333 |
| want | 2.120504 | 8.335341 |
| hate | 2.113494 | 8.277107 |

Since depression is a very tough topic to discuss, most people who have it or know some who have it tend to not understand how to help. It is sometimes difficult for people to reach out for support to deal with their depression so posting online is an outlet for them to let out their emotions. 

In my model, the specificity and sensitivity were the most consistent as well as the best in reducing false positives. I did not want a model that did significantly better than the other because that would could create real world implications. I feel that my model is accurate enough to determine depression thoughts through text so that the "poster" can get the help they need.

## Conclusion and Recommendations

My model was about 91% accurate in determining if a post was from the `r/depression` subreddit or not. If the goal was to beat the baseline score of 50% then I accomplished that, however, there is still about 9% of misclassifications. Looking at the most frequent word/word pairings and coefficients of the top depression word(s), the trend almost matches. A clear distinction is the use of the word "depress" classified a post coming from `r/depression` but how about words that didn't use that word or any variant of it?

This model is in no way perfect but can be a stepping stone, or template, to distinguish random thoughts from true intent/thoughts of depression. The goal of this model is to determine if someone is posting in the depression subreddit or shower thoughts subreddit. If this model were to be improved upon, it would take into consideration dropping a lot of words (stop words) that don't hold much significant meaning, since my best parameters included them. If I could have scraped more posts, maybe then I could have also improved my model while taking into account overfitting. Another thing I wish I had a better grasp on was analyzing the results produced from the SentimentAnalyzer and finding a probable correlation of the composite score with the coefficients of the features.

If given more time, I would have utilized more bootstrapped and bagging methods to check the accuracy of my model. I would also want to see a complete coefficients list of every word in the data and determine the meaning and the weight of the word.

I want to do more research on depression and have a better background in the subject. I feel that in person interaction is the best way to detect if someone is depressed or not as well as just asking questions. I would also like to know the wide scope of how language and writing and how emotions are portrayed. 

I think this type of work could be applicable to clinics or institutions dedicated to giving support and helping those who have depression. Giving them more ways to understand another person's mental state is beneficial for those affected.

---
### Resources

- Raypole, Crystal. “How to Help a Depressed Friend: 15 Do's and Don'ts.” Healthline, Healthline Media, 29 May 2019, www.healthline.com/health/how-to-help-a-depressed-friend. 
- “Depression Basics.” National Institute of Mental Health, U.S. Department of Health and Human Services, www.nimh.nih.gov/health/publications/depression/index.shtml. 
