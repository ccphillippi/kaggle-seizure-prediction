import zipfile
from os.path import join
from contextlib import contextmanager
from collections import OrderedDict
import plac


def ensurePath(path):
    import errno
    from os import makedirs

    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


@contextmanager
def temporary_dir(base_dir):
    from os import path
    from shutil import rmtree
    from uuid import uuid4

    try:
        success = False
        while not success:
            name = str(uuid4())
            dirname = path.join(base_dir, name)
            ensurePath(path.join(base_dir, name))
            success = True

        yield dirname
    finally:
        if success:
            rmtree(dirname)


def SeizureMatFile(path):
    from os.path import basename
    from scipy.io import loadmat

    _mat = loadmat(path, verify_compressed_data_integrity=False)

    dataStruct = _mat['dataStruct']

    names = dataStruct.dtype.names

    items = []
    for name in names:
        item = dataStruct[name][0][0]
        if item.shape == (1, 1):
            item = float(item)
        items.append((name, item))

    filename = basename(path).split('.mat')[0]
    ijk = filename.split('_')
    i, j = ijk[:2]

    items.extend([('patient', int(i)), ('segment', int(j))])

    if len(ijk) == 3:
        items.append(('class', int(ijk[2])))

    return OrderedDict(items)


def matfile_iterator(raw_path, tmp_dir='/dev/shm'):
    with zipfile.ZipFile(raw_path, 'r') as archive:
        with temporary_dir(tmp_dir) as tmp:
            for archive_file in archive.namelist():
                if archive_file.endswith('.mat'):  # folder
                    archive.extract(archive_file, path=tmp)
                    yield SeizureMatFile(join(tmp, archive_file))


@plac.annotations(
    input_dir=('Input directory', 'option'),
    output_dir=('Output directory', 'option'),
    mode=('Mode {train, test}', 'option'),
    tmp_dir=('Temporary folder used to extract archive', 'option'),
    verbose=('Whether or not to display progress', 'option')
)
def main(input_dir='data/raw', output_dir='data/processed', mode='train',
         tmp_dir='/dev/shm', verbose=True):
    '''
    Reads in data from <input dir> and stores it as:

    If mode == 'train':
        stores data into <output_dir>/train/<patient>/<class>/<segment>.mat
    If mode == 'test':
        stores data into <output_dir>/test/<patient>/<segment>.mat
    '''
    from glob import glob
    from os.path import join, dirname
    from scipy.io import savemat

    assert mode in {'train', 'test'}

    base_path = join(output_dir, mode)

    for archive in glob(join(input_dir, mode + '*.zip')):
        for seizure_matfile in matfile_iterator(archive, tmp_dir=tmp_dir):
            patient = str(seizure_matfile['patient'])
            segment = str(seizure_matfile['segment'])

            if mode == 'train':
                k = str(seizure_matfile['class'])
                path = join(base_path, patient, k, segment + '.mat')
            else:
                path = join(base_path, patient, segment + '.mat')

            ensurePath(dirname(path))
            if verbose:
                print 'Storing: %s' % path
            savemat(path, {'data': seizure_matfile['data']},
                    do_compression=True)

if __name__ == '__main__':
    plac.call(main)
