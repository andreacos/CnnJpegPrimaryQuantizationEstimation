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
import cv2
import math
import utils
import pandas as pd
import numpy as np
import configuration as cfg
from glob import glob
from utils import zigzag_order_arr
import random


def multi_to_single_csv(csv_dir, outname):

    folders = glob(csv_dir + '/*/')

    all_files = []
    for f in folders:
        ls = glob(f + '/*.csv')
        if len(ls) > 0:
            all_files.append(ls[0])

    df = pd.concat(pd.read_csv(f, sep=';', header=None) for f in all_files)
    out_f = os.path.join(csv_dir, outname)
    print('Saving data frame {} to {}'.format(df.shape, out_f))
    df.to_csv(out_f, header=None, sep=';')

    return


def multi_to_single_csv_byqf(csv_dir, outname):

    folders = glob(csv_dir + '/*/')

    all_files = []
    for f in folders:
        for qf_p in cfg.q_factors:
            # There might be several folders of QF pairs, used for other experiments
            # Select only QF pairs that are being used for training/test in configuration.py
            if f.find('{}-{}'.format(qf_p[0], qf_p[1])) != -1:
                print('{}-{}'.format(qf_p[0], qf_p[1]))
                ls = glob(f + '/*.csv')
                if len(ls) > 0:
                    all_files.append(ls[0])

    df = pd.concat(pd.read_csv(f, sep=';', header=None) for f in all_files)
    out_f = os.path.join(csv_dir, outname)
    print('Saving data frame {} to {}'.format(df.shape, out_f))
    df.to_csv(out_f, header=None, sep=';')

    return


def make_folder(root, subfold):

    qf_folder = os.path.join(root, subfold)
    if not os.path.exists(qf_folder):
        os.makedirs(qf_folder)

    return qf_folder


def make_dataset(in_dir, out_dir, csv_file, bsize, file_ext='*', rgb=True, jpeg_qfs=(80, 100), blocks_per_image=1000,
                 max_blocks=1e6, force_jpeg_aligned=False):

    if not file_ext.startswith('.'):
        file_ext = '.' + file_ext

    out_folder = make_folder(root=out_dir, subfold=str(jpeg_qfs[0])+'-'+str(jpeg_qfs[1]))

    images = glob(os.path.join(in_dir, '*' + file_ext))

    csv_companion = os.path.join(out_folder, csv_file)

    with open(csv_companion, 'w') as csv:

        it, count = 0, 0
        for idx, fim in enumerate(images):

            try:
                # Read uncompressed image
                im = cv2.imread(fim, int(rgb))

                # Compress image into buffer, then read it as image
                buffer = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qfs[0]])[1]
                single_jpeg_im = cv2.imdecode(buffer, int(rgb))

                # Read quantization matrix from the file: the label of the classification problem will be the
                # first quantisation matrix
                Q = utils.find_Qmatrix_PIL(buffer)[0]
                if cfg.zig_zag_order:
                    Q = zigzag_order_arr(Q)
                else:
                    Q = np.reshape(Q, (64,))

                # Now the enhanced image is sub-divided into blocks. Find largest size multiple of the chosen block size
                if force_jpeg_aligned:
                    shift_r = 0
                    shift_c = 0
                else:
                    shift_r = random.randint(0, 7)   # np.random.randint(0, 8)
                    shift_c = random.randint(0, 7)   # np.random.randint(0, 8)

                single_jpeg_im = single_jpeg_im[shift_r:, shift_c:]

                buffer = cv2.imencode('.jpg', single_jpeg_im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qfs[1]])[1]
                double_jpeg_im = cv2.imdecode(buffer, int(rgb))

                # Convert to YCbCr, take Luminance
                if cfg.rgb:
                    double_jpeg_im_YCBCR = cv2.cvtColor(double_jpeg_im, cv2.COLOR_BGR2YCrCb)
                    double_jpeg_im = double_jpeg_im_YCBCR[:, :, 0]

                # Divide image into blocks
                multiple_height = int(math.floor(double_jpeg_im.shape[0] / float(bsize[0])) * bsize[0])
                multiple_width = int(math.floor(double_jpeg_im.shape[1] / float(bsize[1])) * bsize[1])

                # Count available blocks in current image, pick up random block indices until max_per_image is reached
                available_blocks = int(np.floor(multiple_height / float(bsize[0]))) * int(np.floor(multiple_width / float(bsize[1])))

                perm = np.random.permutation(available_blocks)[:blocks_per_image]

                # Divide the pristine image and its enhanced version into blocks
                n_blocks_img = 0
                for k in range(0, multiple_height, bsize[0]):
                    for l in range(0, multiple_width, bsize[1]):

                        if n_blocks_img in perm:

                            jpeg_patch = double_jpeg_im[k:k + bsize[0], l:l + bsize[1]]

                            img_file = '{:08d}_{}.png'.format(count, os.path.basename(fim))
                            cv2.imwrite(os.path.join(out_folder, img_file), jpeg_patch, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                            # Write file name and its quantisation matrix (the "label" to csv)
                            img_file = os.path.join(out_folder, img_file)

                            csv.write('{};{};{};{};{};{};{}\n'.format(img_file, jpeg_qfs[0], jpeg_qfs[1],
                                                                      cfg.software,
                                                                      utils.Q2string(Q),
                                                                      shift_c,
                                                                      shift_r))

                            count += 1

                        n_blocks_img += 1

                        # Stop when the desired amount of patches has been created
                        if count == max_blocks:
                            print("Reached {} patches".format(max_blocks))
                            print(' ... Done!')
                            return

            except Exception as e:
                print('***** ERROR: {} -- {}*****'.format(fim, str(e)))
                pass

            if count > 0 and count % 5000 == 0:
                print('{} blocks out of {} so far'.format(count, max_blocks))

            it += 1

    return


if __name__ == '__main__':

    assert len(cfg.q_factors) == len(cfg.n_blocks_train) == len(cfg.n_blocks_test)

    for cnt, qf_pair in enumerate(cfg.q_factors):

        print('Dataset JPEG qf1 = {}, qf2 = {}'.format(qf_pair[0], qf_pair[1]))

        # Train set
        if cfg.make_train:

            make_dataset(in_dir=cfg.input_train_dir, out_dir=cfg.out_train_dir,
                         file_ext=cfg.file_ext, csv_file='train{}-{}.csv'.format(qf_pair[0], qf_pair[1]),
                         rgb=cfg.rgb, bsize=cfg.block_size, jpeg_qfs=qf_pair,
                         blocks_per_image=cfg.max_blocks_img, max_blocks=cfg.n_blocks_train[cnt],
                         force_jpeg_aligned=cfg.force_jpeg_aligned)

        # Test set
        if cfg.make_test:

            make_dataset(in_dir=cfg.input_test_dir, out_dir=cfg.out_test_dir,
                         file_ext=cfg.file_ext, csv_file='test{}-{}.csv'.format(qf_pair[0], qf_pair[1]),
                         rgb=cfg.rgb, bsize=cfg.block_size, jpeg_qfs=qf_pair,
                         blocks_per_image=cfg.max_blocks_img, max_blocks=cfg.n_blocks_test[cnt],
                         force_jpeg_aligned=cfg.force_jpeg_aligned)

    if cfg.make_train:
        multi_to_single_csv_byqf(cfg.out_train_dir, cfg.training_csv)

    if cfg.make_test:
        multi_to_single_csv_byqf(cfg.out_test_dir, cfg.test_csv)
