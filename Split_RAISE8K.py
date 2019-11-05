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

import os
import tqdm
import shutil
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-rdir', '--raisedir', help='RAISE8K directory')
    parser.add_argument('-o', '--outputdir', help='Output directory for splitted RAISE8K')
    parser.add_argument('-res', '--resourcedir', help='Directory where txt lists live')
    parser.add_argument('-copy', '--copy', help='Whether to copy (1) or move images (0)', type=int)

    args = parser.parse_args()

    raise_dir = args.raisedir
    out_dir = args.outputdir
    resource_dir = args.resourcedir
    copy = bool(args.copy)

    # ------------------------------------------------------------------------------------------------------------------
    # Creating output folders
    # ------------------------------------------------------------------------------------------------------------------

    print('Creating folders ... ', end='')
    train_raise_folder = os.path.join(out_dir, 'train')
    test_raise_folder = os.path.join(out_dir, 'test')

    if not os.path.exists(train_raise_folder):
        os.makedirs(train_raise_folder)

    if not os.path.exists(test_raise_folder):
        os.makedirs(test_raise_folder)
    print('  > Done')

    # ------------------------------------------------------------------------------------------------------------------
    # Reading lists
    # ------------------------------------------------------------------------------------------------------------------

    print('Reading lists ... ', end='')

    train_list = os.path.join(resource_dir, 'r8k_train.txt')
    test_list = os.path.join(resource_dir, 'r8k_test.txt')

    assert os.path.exists(train_list) and os.path.exists(test_list)

    # Read train images
    with open(train_list) as f:
        train_images = f.readlines()
    train_images = [x.strip() for x in train_images]

    # Read test images
    with open(test_list) as f:
        test_images = f.readlines()
    test_images = [x.strip() for x in test_images]
    print('  > Done')

    # ------------------------------------------------------------------------------------------------------------------
    # Copying files
    # ------------------------------------------------------------------------------------------------------------------

    print('Copying files ... ')

    transfer_fun = shutil.copy2 if copy else shutil.move

    for idx in tqdm.trange(len(train_images), desc='Retrieving train images from {}'.format(train_list)):
        src = os.path.join(raise_dir, train_images[idx])
        dst = os.path.join(train_raise_folder, train_images[idx])
        transfer_fun(src, dst)
    print('  > Training set done')

    for idx in tqdm.trange(len(test_images), desc='Retrieving test images from {}'.format(test_list)):
        src = os.path.join(raise_dir, test_images[idx])
        dst = os.path.join(test_raise_folder, test_images[idx])
        transfer_fun(src, dst)
    print('  > Test set done')
