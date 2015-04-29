SALICON Saliency Evaluation
===================

Saliency evaluation codes for SALICON dataset.

## Requirements ##
- python 2.7

## Files ##
./
- saliconEvalDemo.py (demo script)
- image2json.py (converts a folder of saliency maps to result JSON)

./annotation
- fixations_val2014.json (SALICON 2014 validation set)
- Visit SALICON [download]() page for more details.

./results
- fixations_val2014_fake_results.json (an example of fake results for running demo)
- Visit SALICON [format]() page for more details.

./saliconeval: The folder where all evaluation codes are stored.
- evals.py: The file includes SALICONEval class that can be used to evaluate results on SALICON.
- auc: AUC evalutation code
- sauc: Shuffled AUC evaluation code
- nss: NSS evaluation code
- cc: CC evaluation code

## References ##

- [SALICON: Saliency in Context](http://www.ece.nus.edu.sg/stfpage/eleqiz/publications/pdf/salicon_cvpr15.pdf)
- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- Metrics code (AUC, Shuffled AUC, NSS and CC) are migrated from the [MATLAB code](https://github.com/cvzoya/saliency/) of the [MIT Saliency Benchmark] (http://saliency.mit.edu/) 

## Developers ##
- Ming Jiang
- Shane Huang

## Acknowledgement ##
- [COCO Consortium](http://mscoco.org/people/)
