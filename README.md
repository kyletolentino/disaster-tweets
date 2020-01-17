# Predicting Disasters through Twitter
This is my first, amateur attempt at machine learning and Natural Lanugage Processing to determine disasters from Twitter. :)

## Why did I do this?
I wanted to try implementing my own model without copying an already-solved problem. After doing some tutorials on Kaggle, I
came across [this](https://www.kaggle.com/c/nlp-getting-started), and I thought it would be an interesting project. 
I just used the datasets they provided for my own use, as the actual competition requires the use of AutoML, and I wanted to try out a Tensorflow/Keras solution.

## Setup
**Built with:**

- tensorflow-gpu 2.0.0
- pandas
- matplotlib / seaborn
- sklearn

Anaconda was used to create a virtual environment (so I didn't mess up my computer), and I ran my code on PyCharm. For demonstrating, I used Jupyter Notebook so the results could be shown.  I also installed ntlk when 
I tried preprocessing text myself, but feel free to use whatever (i.e. spaCy).

## My Journey
My very first attempt at cracking this was to preprocess the tweets and use an embedding like word2vec, but my training loss 
was super high and training accuracy was low. Then I discovered a Tensorflow/Keras guide that provided its own tokenizer, which
motivated me to just create a model and go on from there. Ultimately, I ended up with tf_dataset_RNN and keras_DNN, but 
text_processing is still there in case I ever go back to the manual preprocessing/embedding route. My models only consider the Tweets themselves, not the location nor keywords of those Tweets (eventually, I want to incorporate these into feature columns).

### tf_dataset_RNN
After looking at the official Tensorflow guides, I wanted to try converting Pandas dataframes into a tensorflow.data.Dataset format,
as many of their tutorials did this for an "easy" way to setup input pipelines. While I was able to convert, I ~~didn't~~ 
still don't understand the dimensions I was putting into my model. 

After a *lot* of trial and error, I was able to run my model and eventually reach validation accuracy of around 0.79, but it just did
not feel right. [This xkcd comic](https://xkcd.com/1838/) describes what I'm talking about.

### keras_DNN
Feeling tf_dataset_RNN was sketchy, I decided to keep the same tokenization method from tf_dataset_RNN (i.e. fitting on training
data and getting vocabulary), but instead feed the padded sequences directly into the model. With this method, I knew exactly what
I was putting into the model, and decided to focus on this.

Initially, I tried using a RNN, testing a LSTM and Bidirectional wrapper, but again, did not feel as if I completely understood what I
was doing. Instead, I went back to a simple model of adding Dense layers, which outperformed RNN model. When I first ran this, training
loss and accuracy improved, but validation loss and accuracy did not, which was a huge red flag of ***overfitting***.

## Test
For examples of my models, see:
- [tf_dataset_RNN_demo](https://github.com/kyletolentino/disaster-tweets/blob/master/tf_dataset_RNN_demo.ipynb)
- [keras_DNN_demo](https://github.com/kyletolentino/disaster-tweets/blob/master/keras_DNN_demo.ipynb)
- [Predictions](https://github.com/kyletolentino/disaster-tweets/tree/master/predictions)

tf_dataset_RNN mostly averaged **around 0.78 to 0.80** for validation accuracy, but again, I was not satisfied since I did not 
completely understand why. As for keras_DNN, it started out with an **average of 0.74**, but I was able to bring that up to **an average of 0.79 to 0.80** by attempting to prevent overfitting, such as simplifying network and adding Dropout layers.

See [log.txt](https://github.com/kyletolentino/disaster-tweets/blob/master/log.txt) for my documented progress along the way.

## Reflections
The biggest challenge that I faced was making sure my model did not put out straight garbage, because you could have a great model, but if you input terrible data, you will get terrible predictions (which I experienced firsthand with manually tokenizing and embedding).

Although a validation accuracy of around 0.79 could be improved, I'm pretty happy with the results, as this was my first crack at NLP.
In the future, I would like to implement BERT for embedding words, as [that's what the cool kids are using right now](https://jalammar.github.io/illustrated-bert/).
However, with my current GPU, I would only be able to implement BERT on a cloud platform, as using tensorflow-hub on my desktop crashed. I would also like to look into optimizing hyperparameters, as I tried GridSearch using TensorBoard but I'm currently stuck on how to clear log files.

## Credits
I would like to give credit to the following sites/articles for help and inspiration:

- [Kaggle's courses on Machine Learning and Deep Learning](https://www.kaggle.com/learn/overview)
- [Tensorflow's website](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Article on how to solve 90% of NLP problems](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e)
- [Jeff Heaton's Course on Deep Neural Networks](https://github.com/jeffheaton/t81_558_deep_learning)
- [Kaggle Notebok for Credit Fraud Competition](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

### Helpful Links
Here are some articles that I think would be helpful:

- [SGD vs Adam Optimizer](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/)
- [Understanding BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/)
- [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- [Choosing number of steps per epoch](https://stackoverflow.com/questions/49922252/choosing-number-of-steps-per-epoch)
- [Explaining activation functions](https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e)

** Please leave suggestions on how to improve my models, I would appreciate it! :) **
