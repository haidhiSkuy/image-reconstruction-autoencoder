# Autoencoder
An autoencoder is a type of artificial neural network used for unsupervised learning. 
It is designed to learn efficient representations of the input data, typically by compressing the input into a lower-dimensional latent space and then 
reconstructing the input from this compressed representation.

The network consists of two main parts: an encoder and a decoder.
- Encoder: The encoder takes the input data and maps it to a lower-dimensional representation, often referred to as the latent space or encoding. This step is essentially a compression process where the input data is transformed into a compact representation.
- Decoder: The decoder takes the compressed representation produced by the encoder and attempts to reconstruct the original input data from it. It essentially performs the inverse operation of the encoder, mapping the compressed representation back to the original data space.

During training, the autoencoder learns to minimize the difference between the input and the reconstructed output. This is typically done by optimizing a loss function such as mean squared error (MSE) between the input and the output.
Autoencoders have various applications, including dimensionality reduction, data denoising, etc. They are particularly useful when dealing with high-dimensional data, such as images, where learning meaningful representations can be challenging.

![image](https://github.com/widyamsib/image-reconstruction-autoencoder/assets/118953030/6c51ffb7-80af-4716-990d-623bcacc14de)

## Dataset for Training 
your dataset directory should be like this: 
```bash 
dataset
    ├── x
    │   ├── cat_erased1.jpg
    │   ├── cat_erased2.jpg
    │   └── cat_erased3.jpg
    └── y
        ├── cat1.jpg
        ├── cat2.jpg
        └── cat3.jpg
```

![img](https://github.com/widyamsib/image-reconstruction-autoencoder/assets/118953030/e580c51f-3a29-4b0f-b63b-a7ebaf5c6ea7)


## Training 
```console
root@linux:~$ git clone git@github.com:widyamsib/image-reconstruction-autoencoder.git
root@linux:~$ cd image-reconstruction-autoencoder
root@linux:~$ docker build -t haidhi/autoencoder .
```
after you have the image you can start the training 
```console
root@linux:~$ docker run -it --name trainer -v "path-to-dataset":/app/data/cats/ autoencoder 
```
you can specify the number of epochs, batch size, and learning rate
```console
root@linux:~$ docker run -it --name trainer -v "path-to-dataset":/app/data/cats/ autoencoder --epochs 1 --batch-size 1 --learning-rate 0.5
```

## Get Saved Model
the trainer will save the model with the lowest loss for each epoch. After the training is done, you can copy the checkpoint to your local directory 
```console
root@linux:~$ docker cp trainer:"/app/checkpoint/" "$(pwd)"
```
