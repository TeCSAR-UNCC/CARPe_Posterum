# This file is based on the following git repository: https://github.com/agrimgupta92/sgan

# It downloads the dataset presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

wget -O datasets.zip 'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0'
unzip -q datasets.zip
rm -rf datasets.zip
