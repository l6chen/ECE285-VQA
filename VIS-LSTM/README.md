# VIS + LSTM Model

## Descriptions
This is a Tensorflow implementation of the VIS + LSTM visual question answering model from the paper [Exploring Models and Data for Image Question Answering][1]
by Mengye Ren, Ryan Kiros & Richard Zemel. The model architectures vaires slightly from the original - the image embedding is plugged into the last lstm step (after the question) instead of the first. This VIS+LSTM model is based on the github [Torch implementation of neural-VQA with Lua][2]. 
![Model architecture](http://i.imgur.com/Jvixx2W.jpg)

## Code Organization
```
data_loader.py     -- codes for question & answer extraction and pre-processing
extract_fc7.py     -- codes for images extraction and pre-processing
train.py           -- codes for training
predict.py         -- codes for predicting
evaluate.py        -- codes for evaluation
utils.py           -- codes for reading image file and extract image features
vis_lstm_model.py  -- codes for vis_lstm model
demo.ipynb         -- perform demo for prediction
train.ipynb        -- record training process
```

## Datasets
- You have to download the [VQA][3] train+val images, questions and answers by `sh Data/download_data.sh`. Extract all the downloaded zip files inside the `Data` folder.
- You have to download the [pretrained VGG-16 tensorflow model][4] by `sh Data/download_models.sh`.

## Usage

- Retrieve trained model:
```
If you would like to run the demo, unzip model49.ckpt.zip and model49.ckpt.z01 in Data/Models first. 
```
For MacOS system, refer to [this][5]

- Preparation:
```
1. Download pretrained VGG16 Network, using download link written in the download_models.sh file under Data/.
2. Download train images of which the download links are written in the download_data.sh file under Data/.
3. Download val images using download_test.sh, or upload your own test images.
```

- Question and Answers pre-processing:
```
python data_loader.py
```

- Extract the fc-7 image features using:
```
python extract_fc7.py --split=train
python extract_fc7.py --split=val
```

- <b>Training</b>
  * Basic usage `python train.py`
  * Options
    - `rnn_size`: Size of LSTM internal state. Default is 512.
    - `num_lstm_layers`: Number of layers in LSTM
    - `embedding_size`: Size of word embeddings. Default is 512.
    - `learning_rate`: Learning rate. Default is 0.0002.
    - `batch_size`: Batch size. Default is 200.
    - `epochs`: Number of full passes through the training data. Default is 50.
    - `img_dropout`:  Dropout for image embedding nn. Probability of dropping input. Default is 0.5.
    - `word_emb_dropout`: Dropout for word embeddings. Default is 0.5.
    - `data_dir`: Directory containing the data h5 files. Default is `Data/`.
    - `lstm_direc`: The direction of LSTM, unidirectional or bidirection, it can be `uni` or `bi`. Default is `uni`.
    
- <b>Prediction</b>
  * ```python predict.py --image_path="sample_image.jpg" --question="What is the color of the animal shown?" --model_path = "Data/Models/model2.ckpt"```
  * Models are saved during training after each of the complete training data in ```Data/Models```. Supply the path of the trained model in ```model_path``` option.
  
- <b>Evaluation</b>
  * run `python evaluate.py` with the same options as that in train.py, if not the defaults.

## Implementation Details
- fc7 relu layer features from the pretrained VGG-16 model are used for image embeddings. We have implement l2-normalization on the CNN features.
- Questions are zero padded for fixed length questions, so that batch training may be used. Questions are represented as word indices of a question word vocabulary built during pre processing.
- Answers are mapped to 1000 word vocabulary, covering 87% answers across training and validation datasets.
- The LSTM+VIS model is defined in vis_lstm.py. The input tensors for training are fc7 features, Questions(Word indices upto 22 words), Answers(one hot encoding vector of size 1000). The model depicted in the figure is implemented with 2 LSTM layers by default(num_layers in configurable).

## Results
The model achieved an accuray of 55.8% on the validation dataset after 50 epochs of training across the entire training dataset.


## References
- [Exploring Models and Data for Image Question Answering][1]
- [Torch implementation of VQA with Lua][2]

[1]: http://arxiv.org/abs/1505.02074
[2]: https://github.com/abhshkdz/neural-vqa/
[3]: https://visualqa.org/download.html
[4]: https://github.com/ry/tensorflow-vgg16
[5]: https://superuser.com/questions/365643/how-to-unzip-split-files-on-os-x
