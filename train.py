"""
    2019 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:

    Y.Niu, B. Tondi, Y.Zhao, M.Barni:
    “Primary Quantization Matrix Estimation of Double Compressed JPEG Images via CNN",
    IEEE Signal Processing Letters, 2019, November
    Available on ArXiv: arXiv preprint:1908.04259  https://arxiv.org/abs/1908.04259

"""

import keras
from keras.utils import plot_model
from time import time
import os
import random
import configuration as cfg
import math
from utils import read_dataset, assign_exp_id, plot_average_epoch_loss
from networks import custom_qf_loss
from densenet import DenseNet
from batch import next_batch
import numpy as np
import keras.backend as K


if __name__ == '__main__':

    exp_id_model = assign_exp_id(cfg.q_factors, ['ep-{}-coef-{}'.format(cfg.n_epochs, cfg.max_no_Q_coefs)], 'model')
    exp_id_results = assign_exp_id(cfg.q_factors, ['ep-{}-coef-{}'.format(cfg.n_epochs, cfg.max_no_Q_coefs)], 'results')
    if not os.path.exists(os.path.join('results', exp_id_results)):
        os.makedirs(os.path.join('results', exp_id_results))

    if not os.path.exists(os.path.join('models', exp_id_model)):
        os.makedirs(os.path.join('models', exp_id_model))

    # -------------- NEW MODEL FROM SCRATCH IS DONE HERE ----------------------------------------
    model, _ = DenseNet(input_shape=(cfg.block_size[0], cfg.block_size[1], 1), nb_classes=cfg.max_no_Q_coefs)

    # Display / draw information
    model.summary()
    plot_model(model, to_file='models/model.png', show_shapes=True)
    opt = keras.optimizers.adam(lr=cfg.base_lr)

    # -------------- CHOOSE THE LOSS FUNCTION  --------------
    # model.compile(loss=keras.losses.mean_squared_error, optimizer=opt, metrics=["accuracy"])
    # model.compile(loss=custom_qf_loss, optimizer=opt, metrics=["accuracy"])
    model.compile(loss=keras.losses.logcosh, optimizer=opt, metrics=["accuracy"])

    print('The selected training loss is: {}'.format(model.loss.__name__.upper()))

    # Read image paths and labels (i.e. quantisation matrix coefficients) from CSV
    train_images, train_labels, _ = read_dataset(csv_file=os.path.join(cfg.out_train_dir, cfg.training_csv))

    # Determine the number of iterations that complete each epoch (i.e. when the net has seen all the training set)
    max_iterations = int(math.floor(len(train_images) / cfg.batch_size))

    # Start timing
    begin_time = time()

    # Loop epochs
    losses = []

    for ep in range(cfg.n_epochs):

        # Shuffle train data and labels
        perm = list(range(len(train_images)))
        random.shuffle(perm)
        train_images = [train_images[index] for index in perm]
        train_labels = [train_labels[index] for index in perm]

        # Loop training iterations
        for it in range(max_iterations):

            try:
                it_batch, it_labels, it_files = next_batch(batch_size=cfg.batch_size,
                                                           images=train_images,
                                                           labels=train_labels,
                                                           it=it,
                                                           target_size=cfg.block_size)

                # Perform a single iteration
                metrics = model.train_on_batch(it_batch, it_labels)
                losses.append(metrics[0])

                print('Epoch {} Iter {}/{} - Loss: {:3.4f} - LR: {}'.format(ep, it,
                                                                            max_iterations,
                                                                            metrics[0],
                                                                            K.get_value(model.optimizer.lr)))

            except Exception as ex:
                err_log = open('errors.log', 'a+')
                err_log.write('*******ERROR on batch {}! {}******\n'.format(it, str(ex)))
                err_log.close()

            # Save metrics periodically
            if it > 0 and it % cfg.snapshot_frequency == 0:
                print('Epoch {} Iter {}/{} *** Saving metrics ****'.format(ep, it, max_iterations))
                np.save(os.path.join('results', exp_id_results, 'loss.npy'), losses)

        # Save all final data
        model.save(os.path.join('models', exp_id_model, 'model_ep{}.h5'.format(ep)), True, False)
        np.save(os.path.join('results', exp_id_results, 'loss.npy'), losses)

    elapsed = time() - begin_time
    print('-' * 50)
    print('Training ended after {:5.2f} seconds'.format(elapsed))
    print('-' * 50)

    # Plot iteration loss average
    plot_average_epoch_loss(losses, n_epochs=cfg.n_epochs, exp_identifier=exp_id_results.replace('results', ''))
