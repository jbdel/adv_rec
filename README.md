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

Once finished, you can simply run the training with the following commands. It will perfom 5 trainings with the mentionned parameters:

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

We can evaluate on the different test-sets automatically with a beam-search of 16:

```
output=ckpt
nmtpy translate $(ls ${output}/*.best.meteor.ckpt) -s val,test_2016_flickr,test_2017_flickr,test_2017_mscoco,test_2018_flickr -o ${output} -k 16
```

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


## VQA

This section aims to evaluate the G-WGAN results of VQA.

### Requirements

Python3.6<br/>
torch==1.0.1<br/>
torchvision==0.2.2<br/>
numpy<br/>
pillow<br/>
h5py

### Data
Create a dir name data and ckpt.<br/>
Download the [VQA data](https://www.dropbox.com/s/xt6k7aade4o4xrb/data_emnlp2019.zip?dl=1) and place it in the data folder.<br/>
Download the [visual features](https://www.dropbox.com/s/v0lam4928w3nbbe/val36.zip?dl=1) and place it in data.<br/>
Download the [pretrained model checkpoint](https://www.dropbox.com/s/tgogx7sp90o0dup/model_0.6379.pth.zip?dl=1) and place it in ckpt.<br/>
 
### Inference

Start evaluation by typing the following command :
```
output=ckpt
python main.py  \
--batch_size 128  \
--reconstruction True  \
--output $output \
--adv_mode wgan \
--load_cpu True \
--adv 1 \
--eval True \
--ckpt $(basename $output/model* .pth)
```

You should get an accuracy of 63.79%.<br/>
The whole adversarial implementation is available [here](https://github.com/anon0001/adv_rec/blob/master/VQA/adversarial.py)



