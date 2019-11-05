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

from keras.models import load_model
import configuration as cfg
import os
from batch import evaluate_model
from utils import plot_average_accuracy, rearrange_zigzag_array, read_dataset_wfilter, qf1_qf2_coefficients_map


if __name__ == '__main__':

    # Test model
    model_file = 'models/<EXPERIMENT_ID>/<MODEL_NAME>.h5'

    # Data file
    csv_file = os.path.join(cfg.out_test_dir, cfg.test_csv)

    # Load the table linking each pair of JPEG quality factors to the corresponding Q's coefficients
    qf_map = qf1_qf2_coefficients_map(csv_file=csv_file)

    # Load model
    model = load_model(model_file)

    # Read CSV with test dataset for each (QF1, QF2) pair
    for qf_pair in cfg.q_factors:

        test_images, test_labels, test_jpeg_pairs = read_dataset_wfilter(csv_file=csv_file,
                                                                         qf_filter=qf_pair)

        if len(test_images) == 0:
            print('WARNING! NO RECORD FOR {}, {}'.format(qf_pair[0], qf_pair[1]))
            break

        # Test model performance
        csv_output = 'results/test_results_{}_{}.csv'.format(qf_pair[0], qf_pair[1])
        avg_mse, avg_nmse, test_accuracy, accuracy_matrix, average_error = \
            evaluate_model(model=model,
                           images=test_images,
                           labels=test_labels,
                           qfactors=test_jpeg_pairs,
                           qf_map=qf_map,
                           target_size=cfg.block_size,
                           max_samples=None,
                           csv_companion=csv_output)

        # Plot average accuracy (over all images) for each coefficient
        plot_file_acc = 'results/acc_x_coeff_{}_{}.png'.format(qf_pair[0], qf_pair[1])
        plot_average_accuracy(rearrange_zigzag_array(accuracy_matrix, 8), savefile=plot_file_acc)

        print('-' * 80)
        print('QF1 = {} QF2 = {}'.format(qf_pair[0], qf_pair[1]))
        print('-' * 80)
        print('Test average MSE: {:3.4f}'.format(avg_mse))
        print('Test average normalised MSE: {:3.4f}'.format(avg_nmse))
        print('Test accuracy: {:3.4f}'.format(test_accuracy))
        print('-'*80)
        print('\n')
