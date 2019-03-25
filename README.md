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
- Python 3.7
- pipenv

First, the data needs to be acquired and transformed into the correct format for the word2vec container to be created.

From the Stanford github page, https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors, acquire glove.840B.300d.zip and extract the text file somewhere.

To pre-process the file, from the scripts/generate_docker_dataset directory, execute:

```pipenv install```

```pipenv shell```

We can now run the script to create the pkl file used for the container.

```python generate_pickle_data.py {path_to_glove.txt} glove```

The output file glove.840B.300d.pkl will be required to load the word2vec container.

In the src directory create a 'datasets' folder and move the output pkl file into there, and make sure that the 'W2V_VECTOR_FILE' variable in docker-compose.yml matches the name and directory of the pkl file.

To start the service, execute:

```docker-compose up```

# Contribute
To contribute to this project you can choose an existing issue to work on, or create a new issue for the bug or improvement you wish to make, assuming it's approval and submit a pull request from a fork into our master branch.

Once the validation build has completed successfully and a user from the approved list is satisfied with the code review, your changes will be merged into the master branch.
