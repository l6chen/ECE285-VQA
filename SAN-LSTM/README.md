# Visual Question Answering with SANs model in Tensorflow


This is a Tensorflow implementation of the LSTM + CNN + SANs model from the paper [Stacked Attention Networks for Image Question Answering][1]
. This LSTM + CNN + SANs model is based on the github [Torch implementation of neural-VQA with Lua][2]. 


## Requirements
- Python 3.6+
- [Tensorflow][3]
- [h5py][4]


#### Datasets
- Download the [VQA][5] train+val images, questions and answers using `Data/download_data.sh`. Extract all the downloaded zip files inside the `Data` folder.
- Download the [pretrained VGG-16 tensorflow model][6] using `Data/download_models.sh`.

## Usage

- Preparation:
```
1. Unzip Model19.rar file in Data/Models.
2. Download pretrained VGG16 Network, using download link written in the download_models.sh file under Data/.
3. Download train images of which the download links are written in the download_data.sh file under Data/.
4. Download val images using download_test.sh, or upload your own test images.
```

- Question and Answers pre-processing:
```
python data_loader.py
```
- Extract the cnn-7 image features using:
```
python extract_cnn7.py --split=train
python extract_cnn7.py --split=val
```

- <b>Training</b>
  * Basic usage `python train.py`
  * Options
    - `rnn_size`: Size of LSTM internal state. Default is 512.
    - `num_lstm_layers`: Number of layers in LSTM
    - `embedding_size`: Size of word embeddings. Default is 512.
    - `learning_rate`: Learning rate. Default is 0.001.
    - `batch_size`: Batch size. Default is 200.
    - `epochs`: Number of full passes through the training data. Default is 200.
    - `img_dropout`:  Dropout for image embedding nn. Probability of dropping input. Default is 0.5.
    - `word_emb_dropout`: Dropout for word embeddings. Default is 0.5.
    - `data_dir`: Directory containing the data h5 files. Default is `Data/`.

- <b>Prediction</b>
  * ```python predict.py --image_path="sample_image.jpg" --question="What is the color of the animal shown?"```
  * Models are saved during training after each of the complete training data in ```Data/Models```. Supply the path of the trained model in ```model_path``` option.
  
- <b>Evaluation</b>
  * run `python evaluate.py` with the same options as that in train.py, if not the defaults.

## Implementation Details
- Cnn7 maxpool layer features (7x7x512) from the pretrained VGG-16 model are used for image embeddings.
- Questions are zero padded for fixed length questions, so that batch training may be used. Questions are represented as word indices of a question word vocabulary built during pre processing.
- Answers are mapped to 1000 word vocabulary, covering 87% answers across training and validation datasets.
- The SANs model is defined in vis_lstm_model.py. The input tensors for training are cnn7 features, Questions(Word indices upto 22 words), Answers(one hot encoding vector of size 1000). The model depicted in the figure is implemented with 2 LSTM layers by default(num_layers in configurable).

## Results
The model achieved an accuray of 55.6% on the validation dataset after 43 epochs of training across the entire training dataset.


## References
- [Stacked Attention Networks for Image Question Answering][1]
- [Torch implementation of VQA with Lua][2]

[1]: https://arxiv.org/pdf/1511.02274.pdf
[2]: https://github.com/abhshkdz/neural-vqa/
[3]: https://github.com/tensorflow/tensorflow
[4]: http://www.h5py.org/
[5]: https://visualqa.org/download.html
[6]: https://github.com/ry/tensorflow-vgg16
