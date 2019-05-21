# Introduction 
This is the hu:toma implementation of word2vec, as a service that takes an array of words and returns the vectorisation of those words, or null in the case of an unknown word.

# Getting Started
Once the service has been built and started per the 'Build and Test' section of this Readme, the primary endpoint is '/words' which can be accessed on your local machine from http://localhost:9090/words

An example of this request is:

```{"words" : ["word1", "word2"]}```

With this expected response, in the case that we do not have a vectorisation for 'word2':

```{"vectors":{"word1":[...], "word2":null}}```

With the same payload you can use the '/unk_words' endpoint to discover which words don't have vectorisations stored.

There are additional endpoints '/health', and '/reload', which will return the service health and reload the source data file respectively.

# Build and Test
To run a local build of this project, you will need:
- Docker

First, the data needs to be acquired and transformed into the correct format for the word2vec container to be created. We use pre-trained word vectors, further pickled to speedup the loading process. You can get the pickled files from our Google public bucket, for the following languages:
English: [https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/glove.840B.300d.pkl](https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/glove.840B.300d.pkl)
Spanish: [https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/wiki.es.pkl](https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/wiki.es.pkl)
Italian: [https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/wiki.it.pkl](https://storage.googleapis.com/hutoma-datasets/word2vec_service/v2/wiki.it.pkl)
(_Note that you can only use one language per Word2Vec service, but you can have multiple instances of the service, each one supporting a different language_)

Create a folder `src/datasets` and move the .pkl file into it.
Build the docker container with:
```
cd src
docker build -t word2vec .
```
To run the image, run:
```
docker run \
    -p 9090:9090 \
    -v $(pwd)/tests:/tests -v $(pwd)/datasets:/datasets:ro \
    -e "W2V_SERVER_PORT=9090" -e "W2V_VECTOR_FILE=/datasets/glove.840B.300d.pkl" -e "W2V_LANGUAGE=en"\
    word2vec
```
(for the `W2V_VECTOR_FILE` environment variable, make sure you use the appropriate downloaded .pkl file, and for `W2V_LANGUAGE` the corresponding language)

To check that the service is running, try:
```
curl -vv http://localhost:9090/health
```
And you should get a 200 OK response.

### Extending to use different pre-trained Word2Vec word vectors
To use different languages, or different pre-trained Word2Vec word vectors, you will need to generate the .pkl file in a format the service understands. For this you will need:
- Python 3.7
- pipenv
From the Stanford github page, https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors, acquire glove.840B.300d.zip and extract the text file somewhere.

To pre-process the file, from the scripts/generate_docker_dataset directory, execute:

```pipenv install```

```pipenv shell```

We can now run the script to create the pkl file used for the container.

```python generate_pickle_data.py {path_to_glove.txt} glove```

Now you can use the generated file with Word2Vec.


# Contribute
To contribute to this project you can choose an existing issue to work on, or create a new issue for the bug or improvement you wish to make, assuming it's approval and submit a pull request from a fork into our master branch.

Once the validation build has completed successfully and a user from the approved list is satisfied with the code review, your changes will be merged into the master branch.
