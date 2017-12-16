# VisDA2017 Challenge classification rank 3rd
# Method

As for the Classification Challenge task, we use the transfer learning method  based on the two paper "Learning Transferable Features with Deep Adaptation Networks" and "Deep Transfer Learning with Joint Adaptation Networks".
We find the JMMD loss perform better than the MMD loss with deep networks such as resnet50,resnet101. So, in the test stage, inception_resnet_v2,inception v4,senet_resnext50,senet_resnext101 and senet_resnext151 are chosen as our base networks while the loss function is SoftMaxloss and JMMD loss. The training data is only the source dataset The SoftMaxloss is just for the training data and the JMMD loss is used for fitting the training data distribution and test data distribution. 

Finally, we trained 5 models (inception_resnet_v2 inception v4 senet_resnext50 senet_resnext101 senet_resnext151) pretrained by imagenet classification and finetuned the last fc layer in the training set of the task-cv, avoiding overfitting with data augmentation. The top 1 accuracy of each model is: inception_resnet_v2 0.822478991597 inception v4 0.828781512605 senet_resnext50 0.835084033613 senet_resnext101 0.803571428571 senet_resnext151 0.797268907563 Then we merge the
probability of all the 5 models with differenet weights(1.5 2 2.5 0.8 0.8) to get this final results in the testing set. We also compare with no adaptation, we train the senet_resenet50 network with source dataset, and predict on the test data, the accuracy is just 63.2, much lower than using transfer learning method. 

As a result, by the above experiment, we can evaluate that the adapted models perform much better than the source-only model.

# transfer-caffe

This is a caffe repository for transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks".
- Add `jmmd layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks".
- Add `entropy layer` and `outerproduct layer` described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks".
- Copy `grl layer` and `messenger.hpp` from repository [Caffe](https://github.com/ddtm/caffe/tree/grl).
- Emit `SOLVER_ITER_CHANGE` message in `solver.cpp` when `iter_` changes.

Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

Training Model
---------------

In `models/DAN/amazon_to_webcam`, we give an example model based on Alexnet to show how to transfer from `amazon` to `webcam`. In this model, we insert mmd layers after fc7 and fc8 individually.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model. If the Office dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./build/tools/caffe train -solver models/DAN/amazon_to_webcam/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
```

Parameter Tuning
---------------
In mmd-layer and jmmd-layer, parameter `loss_weight` can be tuned to give mmd loss different weights.

Citation
---------------
    @inproceedings{DBLP:conf/icml/LongC0J15,
      author    = {Mingsheng Long and
                   Yue Cao and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Learning Transferable Features with Deep Adaptation Networks},
      booktitle = {Proceedings of the 32nd International Conference on Machine Learning,
                   {ICML} 2015, Lille, France, 6-11 July 2015},
      pages     = {97--105},
      year      = {2015},
      crossref  = {DBLP:conf/icml/2015},
      url       = {http://jmlr.org/proceedings/papers/v37/long15.html},
      timestamp = {Tue, 12 Jul 2016 21:51:15 +0200},
      biburl    = {http://dblp2.uni-trier.de/rec/bib/conf/icml/LongC0J15},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
    @inproceedings{DBLP:conf/nips/LongZ0J16,
      author    = {Mingsheng Long and
                   Han Zhu and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Unsupervised Domain Adaptation with Residual Transfer Networks},
      booktitle = {Advances in Neural Information Processing Systems 29: Annual Conference
                   on Neural Information Processing Systems 2016, December 5-10, 2016,
                   Barcelona, Spain},
      pages     = {136--144},
      year      = {2016},
      crossref  = {DBLP:conf/nips/2016},
      url       = {http://papers.nips.cc/paper/6110-unsupervised-domain-adaptation-with-residual-transfer-networks},
      timestamp = {Fri, 16 Dec 2016 19:45:58 +0100},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/nips/LongZ0J16},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }

    @article{long2017domain,
      title={Domain Adaptation with Randomized Multilinear Adversarial Networks},
      author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
      journal={arXiv preprint arXiv:1705.10667},
      year={2017}
    }
