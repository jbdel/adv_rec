This github repesitory replicates the results found in the paper :

Visual grounding for multimodal tasks with adversarial feedback

# MMT

Create a 3.6 python environement.
Download the data folder with the following link and place it in MMT/

Go to the MMT folder and type :

```
python setup.py develop
```
It will install the required libraries.
Once finished, you can simply run the training with the following commands. It will perfom 5 trainings and then evaluate on the different test-sets automatically with a beam-search of 16.

```
output=out_waae
for i in {1..1}
do
    nmtpy train -C config/waae.conf \
    model.gradient_penality:10 \
    model.critic:5 \
    model.imagination:True \
    model.loss_imagination:waae \
    model.imagination_factor:0.2 \
    train.save_path:./${output}
done
nmtpy translate $(ls ${output}/pool_multi30k-en-de-bpe10k/*.best.meteor.ckpt) -s val,test_2016_flickr,test_2017_flickr,test_2017_mscoco,test_2018_flickr -o ${output} -k 16
```

Results will be printed on screen. You should approximate these results:

| Tables        | BLEU           | METEOR  |
| ------------- |:-------------:| :-----:|
| Test2016      | 40.66 | 60.06 |
| Test2017     | 34.06      |   54.94 |
| COCO-ambiguous | 31.08      |    50.43 |
| Test2018 | 31.91      |    52.37 |

Test2018 scores wont be evaluated correctly as we dont have the ground truth translations.






