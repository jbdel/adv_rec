This github repesitory replicates the results found in the paper :

Visual grounding for multimodal tasks with adversarial feedback

## MMT

This section aims to replicate the G-WGAN results of MMT.

### Data

Create a 3.6 python environement.
Download the data folder with the following [link](https://www.dropbox.com/s/11m17k30tg88oeo/data_nips2019.zip?dl=1) and place it in MMT/

Go to the MMT folder and type :

```
python setup.py develop
```
It will install the required libraries.

### Training

Once finished, you can simply run the training with the following commands. It will perfom 5 trainings and then evaluate on the different test-sets automatically with a beam-search of 16.

```
output=ckpt
for i in {1..5}
do
    nmtpy train -C config/wgan_gp.conf \
    model.gradient_penality:10 \
    model.critic:5 \
    model.imagination:True \
    model.loss_imagination:wgan_gp \
    model.imagination_factor:0.2 \
    train.save_path:./${output}
done
```
### Inference

From this training, we can now perform inference. You can also download a pretrained model (the one used for the challenge submission) [here](https://www.dropbox.com/s/n0v49r93oz0x36q/ckpt_nips2019.zip?dl=1).

Results will be printed on screen. You should approximate these results:

| Tables        | BLEU           | METEOR  |
| ------------- |:-------------:| :-----:|
| Test2016      | 40.66 | 60.06 |
| Test2017     | 34.06      |   54.94 |
| COCO-ambiguous | 31.08      |    50.43 |
| Test2018 | 31.91      |    52.37 |

Test2018 scores wont be evaluated correctly as we dont have the ground truth translations.
You can have more insight of the hyper parameters in the [config file](https://github.com/anon0001/adv_rec/blob/master/MMT/config/wgan_gp.conf). 
You can also check the declaration of both the discriminator and generator used [here](https://github.com/anon0001/adv_rec/blob/master/MMT/nmtpytorch/layers/decoders/conditional.py#L133).






