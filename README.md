# nnio.l
A python package that you can use to create neural networks with one line of code.

## Requirements:
Tensorflow==2.4.0\
scikit-learn==0.24.0\
opencv-python

## Supported architectures:
1. Multilayer Perceptron [Image classification]

## Dataset format:
Dataset\
&nbsp;&nbsp;|__LABEL 1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG n\
&nbsp;&nbsp;|__LABEL 2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG n\
&nbsp;&nbsp;.\
&nbsp;&nbsp;.\
&nbsp;&nbsp;.\
&nbsp;&nbsp;|__LABEL n\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG 2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__IMG n
  

## Example [Creating and training a new MLP]:
```
import nniol

nn = DenseNet(use_pretrained_model=False, path_of_dataset='<PATH OF DATASET HERE>', neurons_per_layer=[<LIST OF INTEGERS SPECIFYING THE NUMBER OF NEURONS IN EACH LAYER>], activations=[<LIST OF STRINGS SPECIFYING ACTIVATION FUNCTIONS FOR EACH LAYER>], model_path='<PATH TO SAVE MODEL>', epochs=<NUMBER OF EPOCHS TO TRAIN>)

nn.predict('<PATH OF DATA TO PASS FOR INFERENCE>')
```
## Example [USING A PRETRAINED MLP]:
```
import nniol

nn = DenseNet(use_pretrained_model=True, model_path='<PATH OF SAVED MODEL>')
nn.predict('<PATH OF DATA TO PASS FOR INFERENCE>')
```
