# Universal Style Transfer via Feature Transforms with TensorFlow & Keras

This is a TensorFlow/Keras implementation of [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf) by Li et al. The core architecture is an auto-encoder trained to reconstruct from intermediate layers of a pre-trained VGG19 image classification net. Stylization is accomplished by matching the statistics of content/style image features through the [Whiten-Color Transform (WCT)](https://www.projectrhea.org/rhea/index.php/ECE662_Whitening_and_Coloring_Transforms_S14_MH), which is implemented here in both TensorFlow and NumPy. No style images are used for training, and the WCT allows for 'universal' style transfer for arbitrary content/style image pairs.

As in the original paper, reconstruction decoders for layers `reluX_1 (X=1,2,3,4,5)` are trained separately and then hooked up in a multi-level stylization pipeline in a single graph. To reduce memory usage, a single VGG encoder is loaded up to the deepest relu layer and is shared by all decoders.

See [here](https://github.com/Yijunmaverick/UniversalStyleTransfer) for the official Torch implementation and [here](https://github.com/sunshineatnoon/PytorchWCT) for a PyTorch version.


## Samples

<p align='center'>
  <img src='samples/gilbert.jpg' width='350px'>
  <img src='samples/gilbert_stylize.png' width='768px'>
</p>


## Requirements

* Python 3.x
* tensorflow 1.2.1+ (it's suggested to have tensorflow-gpu)
* keras 2.0.x
* ~~torchfile~~ Modified torchfile.py is included that is compatible with Windows 
* scikit-image

Optionally:

* OpenCV with contrib modules (for `webcam.py`)
  * MacOS install http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  * Linux install http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
* ffmpeg (for video stylization)


## Running a pre-trained model without gui

1. Download VGG19 model: `bash models/download_vgg.sh`

2. Download checkpoints for the five decoders: `bash models/download_models.sh`

3. Obtain style images. Two good sources are the [Wikiart dataset](https://www.kaggle.com/c/painter-by-numbers) and [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/).


   `python run.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --style-size 512 --alpha 0.8 --style-path /path/to/styleimgs` 

The args `--checkpoints` and `--relu-targets` specify space-delimited lists of decoder checkpoint folders and corresponding relu layer targets. The order of relu targets determines the stylization pipeline order, where the output of one encoder/decoder becomes the input for the next. Specifying one checkpoint/relu target will perform single-level stylization.

Other args to take note of:

* `--style-path`  Folder of style images or a single style image 
* `--style-size`  Resize small side of style image to this
* `--crop-size`  If specified center-crop a square of this size from the (resized) style image
* `--alpha`  [0,1] blending of content features + whiten-color transformed features to control degree of stylization
* `--passes`  # of times to run the stylization pipeline
* `--scale`  Resize content images by this factor before stylizing
* `--keep-colors`  Apply CORAL transform to preserve colors of content
* `--device`  Device to perform compute on, default `/gpu:0`
* `--concat`  Append the style image to the stylized output
* `--noise`  Generate textures from random noise image instead of webcam
* `--random`  Load a new random image every # of frames
* `--adain`  Use [Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) as transfer op instead of WCT

There are also a couple of keyboard shortcuts:

* `r`  Load random image from style folder
* `w`  Write frame to a .png
* `c`  Toggle color preservation
* `s`  Toggle [style swap](#style-swap) (only applied on layer relu5_1)
* `a`  Toggle AdaIN as transform instead of WCT
* `q`  Quit cleanly and close streams

`stylize.py` will stylize content images and does not require OpenCV. The options are the same as for the webcam script with the addition of `--content-path`, which can be a single image file or folder, and `--out-path` to specify the output folder. Each style in `--style-path` will be applied to each content image. 

## Running a pre-trained model with gui

1. Download VGG19 model: `bash models/download_vgg.sh`

2. Download checkpoints for the five decoders: `bash models/download_models.sh`

3. Obtain style images. Two good sources are the [Wikiart dataset](https://www.kaggle.com/c/painter-by-numbers) and [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/).


   `python run_with_gui.py` 
 Advantages of gui:
 * The one doesn't need to reload the programm, so that it's easily to see the difference between results using different learning rate and different layers
 * An efficient way of presenting this technology for the people not familiar with machine learning
 Disadvantages :
 * Because of a big load on processor it's hardly recommended to use a cpu version of tensorflow, if you have gpu with at least 4 GB of memory , then use tensorflow-gpu


## Training decoders

1. Download [MS COCO images](http://mscoco.org/dataset/#download) for content data.

2. Download VGG19 model: `bash models/download_vgg.sh`

3. Train one decoder per relu target layer. E.g. to train a decoder to reconstruct from relu3_1:

   `python train.py --relu-target relu3_1 --content-path /path/to/coco --batch-size 8 --feature-weight 1 --pixel-weight 1 --tv-weight 0 --checkpoint /path/to/checkpointdir --learning-rate 1e-4 --max-iter 15000`

4. Monitor training with TensorBoard: `tensorboard --logdir /path/to/checkpointdir`





## Notes

* This repo is based on [eridgd implementation](https://github.com/eridgd/AdaIN-TF/) of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) by Huang et al. The [AdaIN op](https://github.com/eridgd/WCT-TF/blob/8d9aa9e1c90f91494c45f98c21d0142651d0d669/ops.py#L280-L294) is included here as an alternative transform to WCT. It generally requires multiple stylization passes to achieve a comparable effect.
* The gui support was added by [HikkaV](https://github.com/HikkaV)
* `coral.py` implements [CORellation ALignment](https://arxiv.org/abs/1612.01939) to transfer colors from the content image to the style image in order to preserve colors in the stylized output. The default method uses NumPy and there is also a commented out version in PyTorch that is slightly faster.
* WCT involves two tf.svd ops, which as of TF r1.4 has a GPU implementation. However, this appears to be 2-4x slower than the CPU version and so is explicitly executed on `/cpu:0` in ops.py. [See here](https://github.com/tensorflow/tensorflow/issues/13603) for an interesting discussion of the issue.
* There is [an open issue](https://github.com/tensorflow/tensorflow/issues/9234) where for some ill-conditioned matrices the CPU version of tf.svd will ungracefully segfault. Adding a small epsilon to the covariance matrices appears to avoid this without visibly affecting the results. If this issue does occur, there is a [commented block](https://github.com/eridgd/WCT-TF/blob/5feea790c0d8ca8dc0ffab5e4ec4664045e7084c/ops.py#L55-L62) that uses np.linalg.svd through tf.py_func. This is stable but incurs a 30%+ performance penalty.






