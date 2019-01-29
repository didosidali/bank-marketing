from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

URLBASE = 'https://dl.dropboxusercontent.com/s/{}'
DATA_ROUTS = [
    'zotsgf22f0xr4y0/data-test.csv?dl=0', 'c33xgr93c0hc6e4/data-train.csv?dl=0']
LABELS_ROUTS = [
    'bcl7p7bufqjcwvw/label_test.npy?dl=0', 'mdor7eyqo7l3iy4/label_train.npy?dl=0']

DATA_NAMES = [
    'data-test.csv', 'data-train.csv']
LABELS_NAMES = [
    'label_test.npy', 'label_train.npy']

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