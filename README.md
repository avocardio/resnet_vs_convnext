# Bird Species with Data Distillation

![Bird](https://storage.googleapis.com/kagglesdsdata/datasets/534640/3180974/valid/ALTAMIRA%20YELLOWTHROAT/3.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220214%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220214T142034Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=98f9aaf36ecb59c21663d27553979f90eb8911e241a6226f52c3dde0a7f41f2ff31b2c979d500103f24e1c9d891462491a0e92a302428335993752055dea39964a5a155b53eb155e9c3ef58a07f0fd4eb943b77d87d8ad85b4f94a2b013a3c21c5704bee86736d8b16f0cf1bb2a59642c357156b8127a911bc9e1400141e6f1ffda87fd8552917a797249b7131b92c9d0ab3ab694331e391eed6d198a47936074773dcb60878d87f270ab14848affb229603b2fb33b2b07a94d5167e12acf26226aefdd3d956deb970d2b647e351961ca044756cd86c6de5686e297bb89e58ace1588311cd7bff8ca7b9d3ad02f5b0959632d8dffaafbd0cf01377a36c2267dd)

This is the final project for the course 'Intro to ANNs with Tensorflow'.

## Description

Our project is to use a Lightweight* Convolutational Neural Network to classify images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. The dataset contains 385 different types of birds, and each image is a 224x224x3 colored pixel image. The images are stored in the `data/` folder.

We want to use 2 types of preprocessing techniques, namely data augmentation and *data distillation*. And finally compare results.

Data distillation is a fairly new technique used to improve the performance of a model by reducing required memory and compute. The idea is to use a 'distilled' dataset, containing a subset of the real dataset, with one distilled image per class (i.e. one image per bird). The model is then trained on the distilled dataset, and the model is then tested on the real dataset. The model is then used to classify the real images, without having used the large original dataset.



