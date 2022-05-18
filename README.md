# DCGAN_GENERATE_FACE
采用数据集：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


GAN生成人脸，做了很多版本，这是效果最好的了，但是生成图片的质量任然堪忧
loss 如图，到最后生成器基本不会学习了，



判别器还是学的太快，于是调大了辨别器dropout的概率，但效果非常有限,继续训练下去意义不大，停了



