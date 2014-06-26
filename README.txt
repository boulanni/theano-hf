
I wrapped my Hessian-free code in a generic class, usable as a black-box to train your models if you can provide the cost function as a Theano expression.

It includes all the details in Martens (ICML 2010) and Martens & Sutskever (ICML 2011) crucial to make it work:
- Tikhonov damping with the Levenberg-Marquardt heuristics,
- Gauss-Newton matrix products (you specify an Theano expression `s` to section your computational graph in 2),
- Proper handling of batches and mini-batches (an example SequenceDataset class is provided for variable-length input)
- Conjugate gradient (CG) with information sharing, backtracking, preconditioning and terminations conditions.
- Structural damping for RNNs.

It relies heavily on the Rop. In practice, I could make it work without hassle for a feed-forward network, an RNN with different objectives, NADE (Larochelle) and a more complex model (RNN-NADE) that ties two scans together, so it seems reasonably flexible.
Only the gradients and Gauss-Newton matrix products (95% of the computation) are in Theano, CG and the training logic is in python. It runs on GPU, but for the models I tried, it was a bit slower.
Hessian-free is slow, you need CG batch sizes of 1000+ (don't skimp on this), but you can get really better results than SGD from it with almost zero tweaking.

There is an option to save and recover a checkpoint of training and do early stopping.

I included an RNN example that can memorize an input for 100 time steps (example_RNN). Launch it on 4 cores, come back in 8 hours, and you should have at least one nice solution with 0 error on the validation set.
In comparison, SGD can solve this problem about 0.0% of the time.

It is available here:
https://github.com/boulanni/theano-hf

If you use this software for academic research, please cite the following paper:

[1] N. Boulanger-Lewandowski, Y. Bengio and P. Vincent, "Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription", Proc. ICML 29, 2012.

Author: Nicolas Boulanger-Lewandowski
University of Montreal, 2012
