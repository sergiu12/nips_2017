# nips_2017
simple submission to the nips 2017 kaggle competition

This repository contains the defense 'eairv2p' and the attack 'hpm'
Both the defense and the attack are a slight modification of the samples appearing 
in cleverhans repository: https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/README.md

eairv2p is a modification of: https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/sample_defenses/ens_adv_inception_resnet_v2

hpm is a modification of: https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/sample_attacks/noop

The basic idea behind both defense and attack is to use gaussian_filter to make changes to the original image.

In the defense scenario, sharpened and blurred copies of the input image are produced and fed together with the original image to the ens_adv_inception_resnet_v2 model, the output label is a voting of the three results, if there is a disagreement the predicted label for the original image is used.

In the attack scenario the image is going through a high-pass filter to detect the regions of high contrast, these regions are used to create a mask, and the max allowed perturbation is applied to these areas so as to lower the contrast. 

This code can be ran on a suitable docker, to do so please follow the instructions in the 'Getting Started' section of the kaggle compettion:
https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack#Getting%20started
and the README.md from https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/README.md

Both defense 'eairv2p' and the attack 'hpm' folders can be added to the suitable folders (i.e. sample_defenses and sample_attacks) in the development toolkit supplied by the organisers of the nips+kaggle competition. And should work with the run_attacks_and_defenses.sh script.

