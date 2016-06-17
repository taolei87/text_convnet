
#### News
The code has been re-factored and integrated into the new repo: https://github.com/taolei87/rcnn/tree/master/code/sentiment

The new repo is recommended because it is more modular and supports more running options, type of models etc. 

## CNNs with non-linear and non-consecutive feature maps


This repo contains an implementation of CNNs described in the paper [Molding CNNs for text: non-linear, non-consecutive convolutions](http://arxiv.org/abs/1508.04112) by Tao Lei, Regina Barzilay and Tommi Jaakkola.



#### Dependencies

 * [Theano](http://deeplearning.net/software/theano/) >= 0.7
 * Python >= 2.7
 * Numpy



#### Data
 
 * [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html): <br>
  The annotated constituency trees from the treebank are converted into plain text sequences. `data/stsa.fine.*` are annotated with 5-class fine-grained labels; `data/stsa.binary.*` are annotated with binary labels. <br>
  In the training files `data/stsa.binary.phrases.train` and `data/stsa.fine.phrases.train` we also add phrase annotations. Each phrase (and its sentiment label) is a separate training instance.

 * [Glove](http://nlp.stanford.edu/projects/glove/) word embeddings: <br>
  We use the 840B Common Crawl version. Note the original compressed file is 2GB. In the directory `word_vectors/` we provide a smaller version (~37MB) by pruning words not in the sentiment treebank.

 * [Sogou Chinese news corpora](http://www.sogou.com/labs/dl/c.html): <br>
  Data redistribution is not allowed. Please contact Sogou Lab to obtain the news corpora.


#### Results

Directory `trained_models` contains saved models of the sentiment classification task. We reproduced the results mentioned in our paper by setting the random seed explicitly. The performance of each trained models are listed below:  

Fine-grained models  |  Dev acc.  |  Test acc. 
:--- | --- | ---
stsa.root.fine.pkl.gz  |  49.5  | 50.6 
stsa.phrases.fine.pkl.gz  |  53.4  | 51.2  
stsa.phrases.fine.2.pkl.gz **  |  53.5  |  52.7
| |
**Binary models**  |  **Dev acc.**  |  **Test acc.**
stsa.root.binary.pkl.gz  |  87.0  |  87.0
stsa.phrases.binary.pkl.gz  |  88.9  |  88.6
| |
** **Note**: more recent run (`stsa.phrases.fine.2.pkl.gz`) gets better results than those reported in our paper.

<br>

#### Usage

Our model is implemented in `model.py`. The command `python model.py --help` will list all the parameters and corresponding descriptions.

Here is an example command to train a model on the binary sentiment classification task:
```
python model.py --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --train data/stsa.binary.phrases.train  \
    --dev data/stsa.binary.dev  --test data/stsa.binary.test  \
    --model output_model
```

We can optionally specify Theano configs via `THEANO_FLAGS`:
```
THEANO_FLAGS='device=cpu,floatX=float32'; python model.py ...
```

Another example with more hyperparamter settings:
```
export OMP_NUM_THREADS=1;   #specify number of cores 

THEANO_FLAGS='device=cpu,floatX=float64'; python model.py  \
    --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --train data/stsa.binary.phrases.train  \
    --dev data/stsa.binary.dev  --test data/stsa.binary.test  \
    --model output_model  \
    --depth 3  --order 3  --decay 0.5  --hidden_dim 200  \
    --dropout_rate 0.3  --l2_reg 0.00001  --act relu  \
    --learning adagrad  --learning_rate 0.01
```
