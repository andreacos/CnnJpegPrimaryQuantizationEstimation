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

import pandas as pd
import numpy as np
import io
from wand.image import Image
from matplotlib import pyplot as plt
import cv2
import configuration as cfg


def read_dataset_wfilter(csv_file, qf_filter):

    """ Read the dataset information from input CSV with fields
        | Image paths | JPEG QF1| JPEG QF2 | Quantisation matrix array as string |
        and applies a filter to dataframe rows
        Arguments:
            csv_file    : input csv
            qf_filter   : to-be-applied filter
        Returns:
            files       : list of image paths
            labels      : list of image labels
            jpeg_pairs  : list of (QF1, QF2) for the images
    """

    dataset = pd.read_csv(csv_file, sep=';', header=None,
                          names=['idx', 'filenames', 'quality1', 'quality2', 'software', 'labels', 'shift_r', 'shift_c'])

    # Filter rows by (QF1, QF2)
    isQF1 = dataset['quality1'] == qf_filter[0]
    isQF2 = dataset['quality2'] == qf_filter[1]
    dataset = dataset[isQF1 & isQF2]

    # Read file paths, labels and jpeg pairs
    files = list(dataset['filenames'].values)
    labels = list(dataset['labels'].values)
    jpeg_pairs = list(zip(dataset['quality1'].values, dataset['quality2'].values))

    jpeg_pairs = [(None, int(x[1])) if np.isnan(x[0]) else (int(x[0]), int(x[1])) for x in jpeg_pairs]

    print('Found {} images and {} labels in {} with filter {}'.format(len(files), len(labels), csv_file, qf_filter))

    return files, labels, jpeg_pairs


def read_dataset(csv_file):

    """ Read the dataset information from input CSV with fields
        | Image paths | JPEG QF1| JPEG QF2 | Quantisation matrix array as string |
        Arguments:
            csv_file    : input csv
        Returns:
            files       : list of image paths
            labels      : list of image labels
            jpeg_pairs  : list of (QF1, QF2) for the images
    """

    dataset = pd.read_csv(csv_file, sep=';', header=None,
                          names=['idx', 'filenames', 'quality1', 'quality2', 'software', 'labels', 'shift_r', 'shift_c'])

    files = list(dataset['filenames'].values)
    labels = list(dataset['labels'].values)
    jpeg_pairs = list(zip(dataset['quality1'].values, dataset['quality2'].values))

    jpeg_pairs = [(None, int(x[1])) if np.isnan(x[0]) else (int(x[0]), int(x[1])) for x in jpeg_pairs]

    print('Found {} images and {} labels in {}'.format(len(files), len(labels), csv_file))
    return files, labels, jpeg_pairs


def qf1_qf2_coefficients_map(csv_file):

    """ Creates a table where to each (QF1, QF2) in the input CSV is associated the corresponding
        quantisation matrix

    Keyword arguments:
    csv_file : full path to the CSV input file
    """

    dataset = pd.read_csv(csv_file, sep=';', header=None,
                          names=['idx', 'filenames', 'quality1', 'quality2', 'software', 'labels', 'shift_r', 'shift_c'])
    df1 = dataset[['quality1', 'quality2', 'labels']]

    # Remove duplicates
    df2 = df1.drop_duplicates()

    return np.array(df2)


def find_Qmatrix_PIL(filename):

    """ Extract JPEG quantization matrix for Luma and Chroma with PIL library.
        Returns NaN if image is not JPEG one or both are not found

    Keyword arguments:
    filename : full path to the JPEG image
    """
    from PIL import Image

    if isinstance(filename, str):
        x = Image.open(filename)
    else:
        x = Image.open(io.BytesIO(filename))

    # PIL extracts quantization matrices that are not visited in zig-zag order so we must do that
    zz = np.array(jpeg_zigzag_order(8)).reshape(64)

    if len(x.quantization) == 1:
        q_luma = np.matrix(x.quantization[0])
        q_chroma = np.nan

        # Zig-zag order
        q_luma_arr = np.squeeze(np.asarray(q_luma[0]))
        q_luma = q_luma_arr[zz].reshape(8, 8)

    elif len(x.quantization) == 2:
        q_luma = np.matrix(x.quantization[0])
        q_chroma = np.matrix(x.quantization[1])

        q_luma_arr = np.squeeze(np.asarray(q_luma[0]))
        q_chroma_arr = np.squeeze(np.asarray(q_chroma[0]))

        # Zig-zag order
        q_luma = q_luma_arr[zz].reshape(8, 8)
        q_chroma = q_chroma_arr[zz].reshape(8, 8)

    return np.array(q_luma), q_chroma


def rearrange_zigzag_array(in_arr, size=8, magnify=1):

    """ Rebuilds a matrix in zig-zag order by starting from an array

    Keyword arguments:
    in_arr : input array
    size : size of the output matrix [size x size]
    magnify: the "zooming" factor of the matrix, only for looking good in figures
    """

    zigzag = np.array(jpeg_zigzag_order(size))
    arr2 = in_arr.flatten()
    assert len(arr2) <= size**2

    output = np.zeros((size, size))

    for k in range(len(arr2)):

        for i in range(size):
            for j in range(size):
                if zigzag[i, j] == k:
                    output[i, j] = arr2[k]

    return np.kron(output, np.ones((magnify, magnify)))


def jpeg_zigzag_order(n):

    """ Zig-zag reordering of [n x n] matrix

    Keyword arguments:
    n : size of the matrix to be rearranged in zig-zag order
    """

    def move(i, j):
        if j < (n - 1):
            return max(0, i - 1), j + 1
        else:
            return i + 1, j

    a = [[0] * n for _ in range(n)]
    x, y = 0, 0
    for v in range(n * n):
        a[y][x] = v
        if (x + y) & 1:
            x, y = move(x, y)
        else:
            y, x = move(y, x)
    return a


def zigzag_order_arr(matrix):

    """ Flatten a matrix in zig-zag order.

    Keyword arguments:
    matrix : input n x m matrix
    """

    assert len(matrix.shape) == 2
    n, m = matrix.shape

    ordering = [[] for i in range(n + m - 1)]
    for i in range(n):
        for j in range(m):
            s = i + j
            if s % 2 == 0:
                ordering[s].insert(0, matrix[i][j])

            else:
                ordering[s].append(matrix[i][j])

    zigzag = []
    for i in range(len(ordering)):
        for j in range(len(ordering[i])):
            zigzag.append(ordering[i][j])

    return np.array(zigzag)


def Q2string(Q):

    """ Converts a matrix to a comma separated string .

    Keyword arguments:
    Q : input matrix

    """

    sq = ''
    for x in np.array(Q).flatten():
        sq += str(x) + ','

    return sq[:-1]


def string2Q(sq, size=(8, 8), flatten=True):

    """ Converts a comma separated string to a matrix.

    Keyword arguments:
    sq : input string
    size : output matrix size
    """

    assert len(sq.split(',')) == size[0]*size[1]

    if flatten:
        return np.array([int(x) for x in sq.split(',')]).flatten()
    else:
        return np.array([int(x) for x in sq.split(',')]).reshape(size)


def find_jpeg_quality_buf(buffer):

    """ JPEG quality estimation from buffer image data.

    Keyword arguments:
    buffer : image buffer

    """

    with Image(blob=buffer) as x:
        return int(x.compression_quality)


def find_jpeg_quality(fname):

    """ Estimate JPEG compression quality factor

    Keyword arguments:
    filename : full path to the JPEG image

    """

    if fname.split('.')[-1].lower() not in ['jpeg', 'jpg']:
        print('Input file must be JPEG!')
        return 1

    with Image(filename=fname) as x:
        return x.compression_quality


def plot_metrics(loss_array, max_iterations, exp_identifier):

    """ Plots training loss over epochs

    Keyword arguments:
    loss_array : array with training losses
    max_iterations: training iterations
    exp_identifier: figure's title

    """

    plt.close('all')
    ls = np.array(loss_array).flatten()
    # plt.plot(ls.flatten()[np.arange(0, ls.shape[0], max_iterations)], '.')
    plt.figure(figsize=(12, 9))
    plt.plot(ls.flatten(), '.')
    plt.title(exp_identifier)
    plt.grid()
    plt.ylabel('Training Loss (MSE) per coefficient')
    plt.xlabel('Iterations {:d} iters = 1 epoch'.format(max_iterations))
    plt.show()
    return


def plot_average_error(arr, savefile='mse_vs_coeffs.png'):

    """ Plots MSE (averaged over all dataset) for each estimated coefficient of the
        quantisation matrix

    Keyword arguments:
    arr : array with test MSE
    savefile: if equal to '', displays plots otherwise path to the saved plot image

    """

    fig, ax1 = plt.subplots(1, 1)
    cax1 = ax1.matshow(arr, cmap='Blues')
    ax1.set_title('Average MSE for each coefficient')
    cbar = fig.colorbar(cax1, ax=ax1, cmap='Blues', fraction=0.046, pad=0.04)

    # # Write values over matrix data
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            c = arr[j][i]
            ax1.text(i, j, '{:3.2f}'.format(c), va='center', ha='center', fontsize=18)

    if savefile != '':
        plt.savefig(savefile, dpi=fig.dpi)
        plt.close(fig)
    else:
        plt.show()

    return


def plot_average_accuracy(arr, savefile='accuracy_vs_coeffs.png'):

    """ Plots exact accuracy (averaged over all dataset) for each estimated coefficient of the
        quantisation matrix

    Keyword arguments:
    arr : array with test accuracy
    savefile: if equal to '', displays plots otherwise path to the saved plot image

    """

    fig, ax1 = plt.subplots(1, 1)
    cax1 = ax1.matshow(arr, cmap='Blues')
    ax1.set_title('Average accuracy for each coefficient')
    cbar = fig.colorbar(cax1, ax=ax1, cmap='Blues', fraction=0.046, pad=0.04)

    # # Write values over matrix data
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            c = arr[j][i]
            ax1.text(i, j, '{:3.2f}'.format(c), va='center', ha='center', fontsize=18)

    if savefile != '':
        plt.savefig(savefile, dpi=fig.dpi)
        plt.close(fig)
    else:
        plt.show()

    return


def assign_exp_id(qfactors, add_info, datatype='model'):
    exp_id = ''
    for qf in qfactors:
        exp_id += '{}-{}_'.format(qf[0], qf[1])
    for info in add_info:
        exp_id += '{}_'.format(info)

    return '{}_{}'.format(datatype, exp_id)[:-1]


def mse_qf1_qf2(qf1, qf2, ncoefs=6):

    """ Computes MSE for a given (QF1, QF2) pair on a given number of coefficients of
    the quantisation matrices of QF1, QF2)

    Keyword arguments:
    qf1 : first quality factor
    qf2 : second quality factor
    ncoefs: number of coefficients on which MSE is computed

    """

    dummy = np.random.randint(0, 255, (64, 64))
    buffer_1 = cv2.imencode('.jpg', dummy, [int(cv2.IMWRITE_JPEG_QUALITY), qf1])[1]
    Q1 = find_Qmatrix_PIL(buffer_1)[0]
    if cfg.zig_zag_order:
        Q1 = zigzag_order_arr(Q1)

    buffer_2 = cv2.imencode('.jpg', dummy, [int(cv2.IMWRITE_JPEG_QUALITY), qf2])[1]

    Q2 = find_Qmatrix_PIL(buffer_2)[0]
    if cfg.zig_zag_order:
        Q2 = zigzag_order_arr(Q2)

    mse = np.sum(np.square(Q1[:ncoefs] - Q2[:ncoefs])) / ncoefs
    return mse


def mse_table_qf1_qf2(step=1, ncoefs=6, printme=True):

    """ Computes MSE for all (QF1, QF2) pairs on a given number of coefficients

    Keyword arguments:
    step : sampling step for QF1 and QF2
    ncoefs: number of coefficients on which MSE is computed
    printme: if True, save an image with matrix MSEs

    """

    mse_table = np.zeros((101, 101))

    dummy = np.random.randint(0, 255, (64, 64))
    for i in range(1, 101, step):
        for j in range(1, 101, step):
            mse_table[i][j] = mse_qf1_qf2(i, j, ncoefs)

    if printme:

        fig, ax1 = plt.subplots(1, 1)
        cax1 = ax1.matshow(mse_table, cmap='viridis')
        ax1.set_title('MSE between QM coefficients of (QF1, QF2)')
        cbar = fig.colorbar(cax1, ax=ax1, cmap='Blues', fraction=0.046, pad=0.04)

        # # Write values over matrix data
        # for i in range(0, mse_table.shape[0], step):
        #     for j in range(0, mse_table.shape[1], step):
        #         c = mse_table[j][i]
        #         ax1.text(i, j, '{:3.2f}'.format(c), va='center', ha='center', fontsize=18)

        plt.savefig('mse_table.png', dpi=fig.dpi)
        plt.close(fig)
        np.save('mse_table.npy', mse_table)

    return mse_table


def plot_average_epoch_loss(loss_arr, n_epochs, exp_identifier, show=True):

    loss_arr = np.array(loss_arr)
    iter_epochs = loss_arr.shape[0] // n_epochs
    me = []
    for i in range(n_epochs):
        ls = loss_arr[i * iter_epochs:(i + 1) * iter_epochs]
        me.append(np.mean(ls))

    if show:
        plt.figure(figsize=(12, 9))
        plt.plot(np.arange(n_epochs), np.array(me), 'o-', markersize=8)
        plt.xticks(np.arange(0, n_epochs, step=1))
        plt.yticks(np.arange(0, np.max(me) + 0.5, step=0.5))
        plt.xlabel('Epoch ({:d} iterations)'.format(iter_epochs))
        plt.ylabel('Average MSE (per coefficient)')
        plt.title(exp_identifier)
        plt.grid()
        plt.show()

    return me


def check_shifts(csv_file):

    shifts = []
    with open(csv_file) as fp:

        for cnt, line in enumerate(fp):
            line = line.replace('\n', '')
            shift_r = line.split(';')[-1]
            shift_c = line.split(';')[-2]
            shift = (int(shift_c), int(shift_r))

        if shift not in shifts:
            shifts.append(shift)
            print('Found new shift: {},{}'.format(shift_c, shift_r))

    print('-'*20)
    print('Found {} shifts'.format(len(shifts)))
    print(shifts)

    return shifts


def print_mse():
    t = np.load('/results/mse_table.npy')
    print(mse_qf1_qf2(85, 88))
    return
