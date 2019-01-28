from __future__ import print_function

import os
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

URLBASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/{}'
filename = 'bank.zip'


def main(output_dir='data'):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    url =  URLBASE.format(filename)
    output_file = os.path.join(output_dir, filename)
    if not os.path.exists(output_file):
        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))
        print("Download finished. Extracting files.")
        zipfile.ZipFile(file=output_file, mode="r").extractall(output_dir)
        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
