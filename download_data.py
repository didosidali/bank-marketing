from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


DATA = [('data_train.csv', 'https://drive.google.com/uc?export=download&id=1OycuGIuFTvD8BVdLC_aaN7HQm6-76ctA'),
	('data_test.csv', 'https://drive.google.com/uc?export=download&id=17GKILvg8jlYgSG46-dbaFpZoiueuo3Oz')]

LABEL = [('label_train.npy', 'https://drive.google.com/uc?export=download&id=1PxJJQabqQf55PvHF1FdsxE5uWl2lcUQJ'),
         ('label_test.npy', 'https://drive.google.com/uc?export=download&id=1vZyW2XFXFnIsl_0r_TqfzT2-2xaX3aSm')]

def main(output_dir='data'):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename, url in DATA + LABEL:
        output_file = os.path.join(output_dir, filename)
	
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
            continue

        print('Downloading from {} ...'.format(url))
        urlretrieve(url, filename=output_file)
        print('=> File saved as {}'.format(output_file))


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
