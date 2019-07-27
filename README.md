# GraphX-Convolution

The official implementation of the ICCV paper "GraphX-convolution for point cloud deformation in 2D-to-3D conversion".

![framework](./imgs/framework.jpg)

## Prerequisite

[Pytorch](https://pytorch.org/get-started/locally/) (>=1.0.0)

[Matplotlib](https://matplotlib.org/)

[TensorboardX](https://github.com/lanpa/tensorboardX)

## Data

The data we used in our experiment provided by [ShapeNet](https://www.shapenet.org/). 
However, for convenience, we used the pre-processed data provided by Pixel2mesh. 
The data can be downloaded from [here](https://github.com/nywang16/Pixel2Mesh).

The train/test split used in the paper is the default split provided by ShapeNet. 
For more convenience, we provide the pre-processed train/test file lists in [data](./data).

To process the data into the format used by the code, execute 

```
cd data
python split.py path/to/the/downloaded/data/folder
```

## Training and testing

### Training

After the database is setup, we are ready to train the model. 
At the root of the project, execute

```
cd src
python train.py path/to/the/downloaded/data/folder
```

To resume a half-done training, simply specify the folder containing the weight file using the checkpoint flag

```
python train.py path/to/the/downloaded/data/folder --checkpoint path/to/the/checkpoint/folder
```

For more options, use ```python train.py -h```.

### Evaluation

After the model is fully trained, to test the model, use

```
python test.py path/to/the/downloaded/data/folder --checkpoint path/to/the/checkpoint/folder
```

The script calculates the Chamfer distance (CD) scores similar to [Pixel2mesh](https://github.com/nywang16/Pixel2Mesh).

## References

In our experiment, we actually used a CUDA implementation of CD available [here](https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension), 
which is much more efficient in terms of memory and speed.
