## Detecting Fake News with Python and Machine Learning ##

### Fake news has become a potential problem in recent years, especially during election periods in various countries around the world, such as the 2020 US election and the 2022 Brazilian election, as well as the period of the COVID-19 pandemic. ###

### But after all, how can we identify such fake news? One of the possible answers is with the use of Machine Learning and Python techniques, making it possible to identify patterns found in fake news and, consequently, delete or limit the reach of such publications. ###

### However, before we move on to the project itself, we should clarify how a module and its functions and classes work. The module in question is SKLearn, a module that integrates widespread Machine Learning algorithms with Python resources. ###

### The train_test_split function is capable of splitting vectors and matrices containing data into other randomly formed vectors and matrices for training and testing the performance of Machine Learning. ###

### The TfidfVectorizer class is capable of converting a collection of documents into a TD-IDF matrix. TD stands for Term frequency, which refers to the number of times a word appears in a document. A high number of appearances of the same word in a single document is a good indicator that this may be fake news, if such a word is a search keyword. IDF stands for Inverse Document Frequency, which relates to the number of times a word appears in two or more documents, if this value is high, it may be that this word is insignificant in the search for fake news. ###

### The PassiveAgressiveclassifier class refers to a collection of algorithms that remain online while constantly learning. ###

### Our object of study is a dataset that contains 7796 news items, and can be found at: https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view ###