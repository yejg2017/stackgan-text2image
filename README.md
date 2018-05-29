# stackgan-text2image
text to image learning

Image Source : [Generative Adversarial Text-to-Image Synthesis][2] Paper

## Requirements
- [TensorFlow][4] 1.0+
- [TensorLayer](https://github.com/zsdonghao/tensorlayer) 1.4+
- [NLTK][8] : for tokenizer

## Datasets
- The model is currently trained on the [flowers dataset][9]. Download the images from [here][9] .Also download the captions from [this link][10]. Extract the archive, copy the ```text_c10``` folder and paste it in ```102flowers/text_c10/class_*```.  

**N.B**  You can downloads all data files needed manually or simply run the downloads.py and put the correct files to the right directories.
```python 
python downloads.py
```

## Neccessary folders
```bash
mkdir checkpoints cvpr2016_flowers 102flowers Data imagenet
```

## Codes
- `downloads.py` download Oxford-102 flower dataset and caption files(run this first).
- `data_loader.py` load data for further processing.
- `train_txt2im.py` train a text to image model.
- `utils.py` helper functions.
- `model.py` models.

## References
- [Generative Adversarial Text-to-Image Synthesis][2] Paper
- [Generative Adversarial Text-to-Image Synthesis][11] Torch Code
- [Skip Thought Vectors][1] Paper
- [Skip Thought Vectors][12] Code
- [Generative Adversarial Text-to-Image Synthesis with Skip Thought Vectors](https://github.com/paarthneekhara/text-to-image) TensorFlow code
- [DCGAN in Tensorflow][3]
