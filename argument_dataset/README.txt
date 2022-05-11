UKP Sentential Argument Mining Corpus
-------------------------------------

The UKP Sentential Argument Mining Corpus includes 25,492 sentences 
over eight controversial topics collected with the Google Search API. 
Each sentence was annotated via crowdsourcing as either a supporting
argument, an attacking argument, or not an argument with respect to
the topic.

Each file (one for each of the eight topics) includes the following
information:

* topic: the topic keywords used to retrieve the documents from Google

* retrievedUrl: the original URL retrieved from Google Search API

* archivedUrl: the URL of the retrieved document from the Wayback
  Machine

* sentenceHash: the MD5 hash of the sentence for ensuring
  reproducibility (see below)

* sentence: field for storing the tokenized sentence

* annotation: the gold-standard annotation of the sentence
              (Argument_against, Argument_for, or NoArgument)

* set: the set assigned during data splitting (train, test, val)


Citation
--------

If you find the data useful, please cite the following paper: 

Christian Stab, Tristan Miller, Benjamin Schiller, Pranav Rai, and Iryna Gurevych. Cross-topic argument mining from heterogeneous sources. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), October 2018.

   @inproceedings{stab2018bcross-topic,
      author       = {Christian Stab and Tristan Miller and Benjamin Schiller and Pranav Rai and Iryna Gurevych},
      title        = {Cross-topic Argument Mining from Heterogeneous Sources},
      booktitle    = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)},
      month        = oct,
      year         = {2018},
   }


Licence
-------

The annotations are released under the 
Creative Commons Attribution-NonCommercial licence (CC BY-NC).
