# RNNs on Cognitive Modelling (Subject-Verb Number Agreement)

Code for the Master's Thesis [Recurrent Neural Networks for Cognitive Modelling]
The study is to find recurrent networks with architectures that are biologically
plausible and have performance similar to humans. (LSTMs have the best performance
similarity but are bad models of the brain architecturally).
In the process we also find significant surprsing observations about the performance
of RNNs on certain tasks with changing difficulty and interesting insights into 
vanishing and exploding gradients which needs to be futher looked into.

Dependencies:

* numpy
* pytorch
* inflect
* pandas

Suggested : Intall Anaconda (python library manager). Then install inflect, pytorch
and any other libraries as needed.

## Quick start

Follow this section if you'd like to run the code on the same set of dependencies used by in the thesis
which is the same as used by Linzen in his paper [Assesing the ability of LSTMs to learn syntax-senstive dependcies].

All of the functions should accept all relevant filenames as arguments, but in
general the easiest thing to do is to set the environment variable
`RNN_AGREEMENT_ROOT` to wherever you cloned this repository.

After cloning the repository, download the [dependency
dataset](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz)
into the `data` subdirectory and unzip the file.

```python
from rnnagr.agreement_acceptor import PredictVerbNumber
from rnnagr import filenames

pvn = PredictVerbNumber(filenames.deps, prop_train=0.1)
pvn.pipeline()
```

After running this code, `pvn.test_results` will be a pandas data frame
with all of the relevant results.

Functions to run experiments other than Number Prediction (PredictVerbNumber) 
can be found in the agreement_acceptor.py file in source. 
There is a sample run_plus.py file provided for running the Plus 10 task.

To change the model you are running, change the import file name in agreement_acceptor file.
Each model class is named LSTM inside different files to support one change run.
