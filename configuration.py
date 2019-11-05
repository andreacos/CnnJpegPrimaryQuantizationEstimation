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

# ----------------------------------------------------------------------------------------------
# Parameters for dataset creation
# ----------------------------------------------------------------------------------------------

rgb = True                     # If TRUE, coefficients come from Y channel in YCbCr, if FALSE from grayscale
block_size = (64, 64)          # Image size for the whole training / testing process
max_blocks_img = 100           # Number of blocks that are created for each image

file_ext = '.TIF'              # File format for input images (RAISE8K has only TIFF images)
make_train = True              # If FALSE, train dataset is not created (useful for testing new QF pairs)
make_test = True               # If FALSE, test dataset is not created

# This is the starting folder, from which Train and Test datasets are created
input_train_dir = '/Datasets/RAISE8Ksplit/Train'
input_test_dir = '/Datasets/RAISE8Ksplit/Test'

# This is the output folder where Train and Test datasets are created
out_train_dir = '/Datasets/DeepQuantiFinder/Train'
out_test_dir = '/Datasets/DeepQuantiFinder/Test'

# These are the CSV used to train and test the model. They live in out_train_dir and out_test_dir
training_csv = 'train.csv'
test_csv = 'test.csv'

# ----------------------------------------------------------------------------------------------
# JPEG parameters
# ----------------------------------------------------------------------------------------------
software = 'python-opencv'              # Software used for JPEG compression
force_jpeg_aligned = False              # If TRUE, only aligned JPEG patches are created
zig_zag_order = True                    # If TRUE, JPEG coefficients are always in zig-zag order

# First case-study QF2 = 90
q_factors = [(60, 90), (65, 90), (70, 90), (75, 90), (80, 90), (85, 90), (90, 90), (95, 90), (98, 90)]
n_blocks_train = [4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5]
n_blocks_test = [4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4]

# Second case-study QF2 = 80
# q_factors = [(55, 80), (60, 80), (65, 80), (70, 80), (75, 80), (80, 80), (85, 80), (90, 80), (95, 80)]
# n_blocks_train = [4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5, 4e5]
# n_blocks_test = [4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4, 4e4]

# ----------------------------------------------------------------------------------------------
# Training / Test parameters
# ----------------------------------------------------------------------------------------------

base_lr = 1e-5                         # Learning rate
max_no_Q_coefs = 15                    # Number of quantisation coefficients used for training
batch_size = 32                        # Training batch size
n_epochs = 40                          # Training epochs
scaling_factor_data = 255.0            # Input images (values [0, 255]) are scaled to [0, 1]
snapshot_frequency = 1000              # Frequency (iterations for saving training metrics)
