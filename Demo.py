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

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from batch import preprocess_input, get_label


def show_labels(gt_label, pred_label, width=0.45):
    """ Shows ground truth and estimated quantization coefficients side-by-side as bars
        Arguments:
            gt_label    : ground truth quantization coefficients
            pred_label  : list of image paths
            width       : bar width
    """

    ind = np.arange(len(gt_label))  # the x locations for the groups

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, true_label, width, color='royalblue')

    estimated = np.round(pred_label)
    rects2 = ax.bar(ind + width, estimated, width, color='orange')

    # add some
    ax.set_ylabel('Coefficient value')
    ax.set_xlabel('Coefficient index (zig-zag)')
    ax.set_title('Ground truth vs estimated')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(ind)

    ax.legend((rects1[0], rects2[0]), ('Ground truth', 'Estimated (rounding)'))

    plt.show()


if __name__ == '__main__':

    # These parameters must correspond to those used for training in configuration.py
    max_no_Q_coefs = 15             # Number of estimated coefficients
    block_size = (64, 64)           # Input image patch size
    scaling_factor_data = 255       # Input image patch size pixel scaling factor

    model_file = 'models/pre_trained/DNN90_60LOG.h5'
    model = load_model(model_file)

    # This image comes from the Test set and has been compressed with (70, 90)
    img_name = 'resources/reproducibility/images/00000000_rfaa15f27t.TIF.png'

    # These are all the 64 quantization coefficients already in zig-zag order for QF1=70
    # The model will estimate the first 15
    img_label = '10,7,7,8,7,6,10,8,' \
                '8,8,11,10,10,11,14,24,' \
                '16,14,13,13,14,29,21,22,' \
                '17,24,35,31,37,36,34,31,' \
                '34,33,38,43,55,47,38,41,' \
                '52,41,33,34,48,65,49,52,' \
                '57,59,62,62,62,37,46,68,' \
                '73,67,60,72,55,61,62,59'

    # Prepare image and label
    x = preprocess_input(img_name, block_size, scaling_factor_data)
    true_label = np.array(get_label(img_label))

    # Predict
    predicted_label = model.predict(np.expand_dims(np.expand_dims(x, -1), 0)).flatten()

    print('True label: {}'.format(true_label))
    print('Predicted label: {}'.format(predicted_label))
    print('MSE = {:3.3f}'.format(np.sum(np.square(true_label - predicted_label.flatten())) / max_no_Q_coefs))
    print('Accuracy = {:3.3f}'.format(1 - np.count_nonzero(np.round(predicted_label) == true_label/max_no_Q_coefs)))
    show_labels(true_label, predicted_label)
