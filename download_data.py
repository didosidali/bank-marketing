from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

URLBASE = 'https://dl.dropboxusercontent.com/s/{}'
DATA_ROUTS = [
    '0jobil7exo46qt9/data_test.csv?dl=1', '735rp4zlxhencju/data_train.csv?dl=1']
LABELS_ROUTS = [
    'jerexe2cu2efqbq/labels_test.npy?dl=1', 'j5g7k19lch05rx9/labels_train.npy?dl=1']

DATA_NAMES = [
    'data_test.csv', 'data_train.csv']
LABELS_NAMES = [
    'labels_test.npy', 'labels_train.npy']

def main(output_dir='data'):
    file_routs = DATA_ROUTS + LABELS_ROUTS
    filenames = DATA_NAMES+ LABELS_NAMES
    urls = [URLBASE.format(filename) for filename in file_routs]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    #test = os.getenv('RAMP_TEST_MODE', 0)

    #if test:
    #    print("Testing mode, not downloading any data.")
    #else:
    main()
