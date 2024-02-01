
from os.path import join, exists, split, isdir, basename
from os import listdir, mkdir, sep, environ, walk
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x

from skimage.measure import block_reduce
from skimage.transform import rescale
from time import sleep
import torch
import numpy as np
import nibabel as nib
import pickle
from torch.nn.functional import interpolate


def read_nii(nii_file, additional_channels=0):
    img = nib.load(nii_file)
    spacing = img.header['pixdim'][1:4]
    im = img.get_fdata()
    has_z_first = _has_z_first(spacing, dims=img.shape, filename=nii_file)
    # now we (hopefully) know if the z axis is first or last
    if not has_z_first:
        # z axis is in the back, get it to the front!
        spacing = np.array([spacing[2], spacing[0], spacing[1]])
        im = np.moveaxis(im, 2, 0)

    if additional_channels > 0:
        add_chan = np.random.randn(additional_channels, *im.shape)
        im = np.vstack([np.expand_dims(im, axis=0), add_chan])

    return im, spacing, has_z_first

def _has_z_first(spacing, dims, filename):
    spacing = np.around(spacing,decimals=2)
    global _isotropic_volume_loaded_warning_printed, _ananisotropic_volume_loaded_warning_printed
    # first check if the z axis is last or first
    if spacing[0] == spacing[1]:
        if spacing[0] != spacing[2]:
            has_z_first = False
        elif dims[0] == dims[1] and dims[0] != dims[2]:
            # in the case of isotropic voxel we check the dimensions of the image instead
            has_z_first = False
        elif dims[1] == dims[2] and dims[0] != dims[2]:
            has_z_first = True
        else:
            if not _isotropic_volume_loaded_warning_printed:
                print('Found at least one file {} with isotropic voxel and equal volume dimensions.'
                      'could not infere if the z axis is first or last, guessing last. '
                      'Please make sure it is!'.format(filename))
            has_z_first = False
            _isotropic_volume_loaded_warning_printed = True
    else:
        # spacing[0] != spacing[1]
        if spacing[1] == spacing[2]:
            has_z_first = True
        elif spacing[0] == spacing[2]:
            has_z_first = False
        else:
            if not _ananisotropic_volume_loaded_warning_printed:
                print('Found at least one file {} with voxelspacing {}. '
                      'Need at least two equal numbers in the spacing to find out if the z '
                      'axis is first or last, guessing last. Please make sure it is!'
                      ''.format(filename, spacing))
            has_z_first = False
            _ananisotropic_volume_loaded_warning_printed = True
    return has_z_first

def maybe_create_path(path):

    if path:
        counter = 0
        subfs = []
        bp = path

        while (not exists(bp)) and counter < 100:
            if bp.find(sep) >= 0:
                bp, subf = split(bp)
                subfs.append(subf)
            else:
                break

        if len(subfs) > 0:

            for subf in subfs[::-1]:
                mkdir(join(bp, subf))
                bp = join(bp, subf)
        else:
            if not exists(bp):
                mkdir(bp)


def load_pkl(path_to_file):
    with open(path_to_file, 'rb') as file:
        data = pickle.load(file)
    return data

def read_data_tpl_from_nii(folder, case,additional_channels = 0):
    data_tpl = {}

    if not exists(folder):
        folder = join(environ['OV_DATA_BASE'], 'raw_data', folder)

    if not exists(folder):
        raise FileNotFoundError('Can\'t read from folder {}. It doesn\'t exist.'.format(folder))

    if isinstance(case, int):
        case = 'case_%03d.nii.gz' % case

    if not isinstance(case, str):
        raise TypeError('Input \'case\' must be string, not {}'.format(type(case)))

    if not case.endswith('.nii.gz'):
        case += '.nii.gz'

    # first let's read the data info
    if exists(join(folder, 'data_info.pkl')):
        data_info = load_pkl(join(folder, 'data_info.pkl'))
        if case[:-7] in data_info:
            # the [:-7] is to remove the .nii.gz
            data_tpl.update(data_info[case[:-7]])

    # first check if the image folder exists
    possible_image_folders = ['images', 'imagesTr', 'imagesTs']
    image_folders_ex = [join(folder, imf) for imf in possible_image_folders
                        if exists(join(folder, imf))]
    if len(image_folders_ex) == 0:
        raise FileNotFoundError('Didn\'t find any image folder in {}.'.format(folder))

    image_files = []
    for image_folder in image_folders_ex:
        matching_files = [join(folder, image_folder, file) for file in listdir(join(folder, image_folder)) if file[:-7]==case[:-7]]
        if len(matching_files) > 0 and len(image_files) > 0:
            raise FileExistsError('Found images for in multiple image folders at path {} for '
                                  'case {}.'.format(folder, case))
        image_files = matching_files

    if len(image_files) == 0:
        raise FileNotFoundError('No image files found for case {}.'.format(case))
    elif len(image_files) == 1:
        raw_image_file = image_files[0]
        im, spacing, had_z_first = read_nii(raw_image_file,additional_channels=additional_channels)
    else:
        print(image_files)
        assert(1==2)
        raw_image_file = image_files
        im_data = [read_nii(file,additional_channels=additional_channels) for file in raw_image_file]
        ims = [im for im, spacing, had_z_first in im_data]
        spacings = [spacing for im, spacing, had_z_first in im_data]
        hzf_list = [had_z_first for im, spacing, had_z_first in im_data]

        if not np.all([np.all(spacings[0] == sp) for sp in spacings[1:]]):
            print(spacings)
            raise ValueError('Found unequal spacings when reading the image files {}'
                             ''.format(image_files))

        if not np.all([np.all(hzf_list[0] == hzf) for hzf in hzf_list[1:]]):
            raise ValueError('Found some files with the z axis first and some with z axis last '
                             'when reading the image files {}'.format(image_files))

        im = np.stack(ims)
        spacing = spacings[0]
        had_z_first = hzf_list[0]
    data_tpl['image'] = im
    data_tpl['spacing'] = spacing
    data_tpl['had_z_first'] = had_z_first
    data_tpl['raw_image_file'] = raw_image_file

    label_folders_ex = [join(folder, lbf) for lbf in ['labels', 'labelsTr', 'labelsTs']
                        if exists(join(folder, lbf))]

    if len(label_folders_ex) == 0:
        # when there are no existing label folders we can just return the data tpl
        return data_tpl

    label_files = []
    for label_folder in label_folders_ex:
        matching_files = [join(folder, label_folder, file) for file in listdir(join(folder, label_folder)) if file[:-7] == case[:-7]]
        if len(matching_files) > 0 and len(label_files) > 0:
            raise FileExistsError('Found labels for in multiple label folders at path {} for '
                                  'case {}.'.format(folder, case))
        label_files = matching_files
    if len(label_files) == 0:
        # in case we don't find a label file let's return without
        return data_tpl
    elif len(label_files) == 1:
        lb, spacing, had_z_first = read_nii(label_files[0])
        if np.max(spacing - data_tpl['spacing']) > 1e-4:
            raise ValueError('Found not matching spacings for case {}.'.format(case))
        if had_z_first != data_tpl['had_z_first']:
            raise ValueError('Axis ordering doesn\'t match for case {}'
                             'Make sure image and label files have the z axis at the same position'
                             '(first or last).'.format(case))

        data_tpl['label'] = lb
        data_tpl['raw_label_file'] = label_files[0]
    else:
        raise FileExistsError('Found multiple label files for case {}'.format(case))

    return data_tpl



class raw_Dataset(object):

    def __init__(self, raw_path, scans=None, image_folder=None, dcm_revers=True,
                 dcm_names_dict=None, prev_stages=None, additional_channels=0,
                 create_missing_labels_as_zero=False):

        assert image_folder in ['images', 'imagesTr', 'imagesTs', None]

        self.raw_path = raw_path
        self.create_missing_labels_as_zero = create_missing_labels_as_zero
        self.additional_channels = additional_channels
        if not exists(self.raw_path):
            p = join(environ['OV_DATA_BASE'], 'raw_data', self.raw_path)
            if exists(p):
                self.raw_path = p
            else:
                raise FileNotFoundError('Could not find {} or {}'.format(p, raw_path))

        all_im_folders = [imf for imf in listdir(self.raw_path) if imf.startswith('images')]
        all_lb_folders = [lbf for lbf in listdir(self.raw_path) if lbf.startswith('labels')]

        # prev_stage shold be a dict with the items 'preprocessed_name', 'model_name', 'data_name'
        self.is_cascade = prev_stages is not None
        if self.is_cascade:
            # if we only have one previous stage we can also input just the dict and not the list
            if isinstance(prev_stages, dict):
                prev_stages = [prev_stages]

            self.prev_stages = prev_stages

            # now let's find the prediction pathes and create the keys for the data_tpl
            self.pathes_to_previous_stages = []
            self.keys_for_previous_stages = []
            for prev_stage in self.prev_stages:
                for key in ['data_name', 'preprocessed_name', 'model_name']:
                    assert key in prev_stage

                p = join(environ['OV_DATA_BASE'],
                         'predictions',
                         prev_stage['data_name'],
                         prev_stage['preprocessed_name'],
                         prev_stage['model_name'])
                key = '_'.join(['prediction',
                                prev_stage['data_name'],
                                prev_stage['preprocessed_name'],
                                prev_stage['model_name']])
                raw_data_name = basename(self.raw_path)
                fols = [f for f in listdir(p) if f.startswith(raw_data_name)]
                if len(fols) == 0 and prev_stage['data_name'] == raw_data_name:
                    fols = ['cross_validation']

                if len(fols) != 1:
                    raise FileNotFoundError('Could not identify nifti folder from previous stage '
                                            'at {}. Found {} folders starting with {}.'
                                            ''.format(p, len(fols), raw_data_name))
                self.pathes_to_previous_stages.append(join(p, fols[0]))
                self.keys_for_previous_stages.append(key)
        self.is_nifti = len(all_im_folders) > 0

        if self.is_nifti:

            if len(all_im_folders) > 1 and scans is None and image_folder is None:
                raise ValueError('Multiple image folders found at {}, but no scans were given '
                                 'neither was image_folder set. If there is more than one folder '
                                 'from [\'images\', \'imagesTr\', \'imagesTs\'] contained '
                                 'please specifiy which to read from or give a list of scans as '
                                 'input to raw_Dataset.')
            elif image_folder is not None:
                assert image_folder in all_im_folders
                self.image_folder = image_folder
            elif len(all_im_folders) == 1 and image_folder is None:
                self.image_folder = all_im_folders[0]

            # now the self.image_folder should be set

            if scans is None:
                # now try to get the scans
                labelfolder = 'labels' + self.image_folder[6:]
                if labelfolder in all_lb_folders:
                    self.scans = [scan[:-7] for scan in listdir(join(self.raw_path,
                                                                     labelfolder))]
                else:
                    self.scans = [scan[:-7] for scan in listdir(join(self.raw_path,
                                                                     self.image_folder))]
            else:
                self.scans = scans
            print('Using scans: ', self.scans)

        else:
            # dcm case
            print('The folder {} was not identified as a nifti folder, assuming dcms are '
                  'contained.'.format(self.raw_path))
            self.dcm_revers = dcm_revers
            self.dcm_names_dict = dcm_names_dict
            if scans is None:
                self.scans = []
                for root, dirs, files in walk(self.raw_path):
                    if len(files) > 0:
                        self.scans.append(root)
            else:
                self.scans = [join(self.raw_path, scan) for scan in scans]

            len_rawp = len(self.raw_path)
            print('Using scans: ', [scan[len_rawp:] for scan in sorted(self.scans)])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, ind=None):

        if self.__len__() == 0:
            return

        if ind is None:
            ind = np.random.randint(len(self.scans))
        else:
            ind = ind % len(self.scans)

        scan = self.scans[ind]

        if self.is_nifti:
            data_tpl = read_data_tpl_from_nii(self.raw_path, scan, additional_channels=self.additional_channels)
        else:
            raise NotImplementedError('dcm reading not implemented yet')

        if 'label' not in data_tpl and self.create_missing_labels_as_zero:
            data_tpl['label'] = np.zeros(data_tpl['image'].shape[-3:])

        if self.is_cascade:
            for path, key in zip(self.pathes_to_previous_stages, self.keys_for_previous_stages):
                pred_fps, _, _ = read_nii(join(path, scan + '.nii.gz'), additional_channels=self.additional_channels)
                data_tpl[key] = pred_fps

        data_tpl['dataset'] = basename(self.raw_path)
        data_tpl['scan'] = scan
        return data_tpl


class SegmentationPreprocessing(object):
    '''
    Class that is responsible for performing preprocessing of segmentation data
    This class expects
        1) single channel images
        2) non overlappting segmentation in integer encoding
    If the corresponding flags are set we perform
         1) resizing to change the pixel spacing to target_spacing
         2) additional downsampling by factor 2, 3, or 4
         3) windowing/clipping of image values
         4) scaling of the gray values x --> (x-scaling[1])/scaling[0]
    Images will be resampled with first or third order by default.
    Segementations are decoded to one hot vectors resampled by trilinear
    interpolation and decoded to integer encoding by argmax
    '''

    def __init__(self,
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=None,
                 lb_classes=None,
                 reduce_lb_to_single_class=False,
                 lb_min_vol=None,
                 lb_max_vol=None,
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 save_only_fg_scans=False,
                 prev_stages=[],
                 num_additional_channels=0,
                 dataset_properties={}):

        # first the parameters that determine the preprocessing operations
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.target_spacing = target_spacing
        self.pooling_stride = pooling_stride
        self.window = window
        self.scaling = scaling
        self.lb_classes = lb_classes
        self.reduce_lb_to_single_class = reduce_lb_to_single_class
        self.lb_min_vol = lb_min_vol
        self.lb_max_vol = lb_max_vol
        self.n_im_channels = n_im_channels
        self.do_nn_img_interp = do_nn_img_interp
        self.prev_stages = prev_stages
        # this is only important for preprocessing of raw data
        self.save_only_fg_scans = save_only_fg_scans
        self.dataset_properties = dataset_properties
        # when inheriting from this please append any other important
        # parameters that define the preprocessing
        self.num_additional_channels = num_additional_channels
        print("num additional channels")
        self.preprocessing_parameters = ['apply_resizing',
                                         'apply_pooling',
                                         'apply_windowing',
                                         'target_spacing',
                                         'pooling_stride',
                                         'window',
                                         'scaling',
                                         'lb_classes',
                                         'reduce_lb_to_single_class',
                                         'lb_min_vol',
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'prev_stages',
                                         'dataset_properties']
        if isinstance(self.prev_stages, dict):
            self.prev_stages = [self.prev_stages]

        assert isinstance(self.prev_stages, list), 'prev_stages must be given as a list or dict'

        if self.is_cascade():
            # creating all the keys for the predictions from the previous stage.
            # we do this here as a double check that when a new data_tpl comes in we will see
            # if all the predictions are here
            self.keys_for_previous_stages = []
            for prev_stage in self.prev_stages:
                for key in ['data_name', 'preprocessed_name', 'model_name']:
                    assert key in prev_stage
                key = '_'.join(['prediction',
                                prev_stage['data_name'],
                                prev_stage['preprocessed_name'],
                                prev_stage['model_name']])

                self.keys_for_previous_stages.append(key)

        self.is_initalised = False

        if self.check_parameters():
            self.initialise_preprocessing()
        else:
            # some parameters are missing
            print('Preprocessing was not initialized with necessary parameters. '
                  'Either load these with \'try_load_preprocessing_parameters\', '
                  'or infere them from raw data with \'plan_preprocessing_from_raw_data\'.'
                  'If you modify these parameters call \'initialise_preprocessing\'.')

    def is_cascade(self):
        return len(self.prev_stages) > 0

    def check_parameters(self):

        if self.scaling is None:
            return False

        if self.apply_resizing and self.target_spacing is None:
            return False

        if self.apply_pooling and self.pooling_stride is None:
            return False

        if self.apply_windowing and self.window is None:
            return False

        return True

    def initialise_preprocessing(self):

        if not self.check_parameters():
            print('Not all required parameters were initialised, can not initialise '
                  'preprocessing objects')
            return

        include_keys = ['apply_resizing', 'apply_pooling', 'apply_windowing', 'target_spacing',
                        'pooling_stride', 'window', 'scaling', 'n_im_channels',
                        'do_nn_img_interp']

        inpt_dict_3d = {key: self.__getattribute__(key) for key in
                        self.preprocessing_parameters if key in include_keys}
        inpt_dict_2d = inpt_dict_3d.copy()
        if self.apply_resizing:
            inpt_dict_2d['target_spacing'] = self.target_spacing[1:]
        if self.apply_pooling:
            inpt_dict_2d['pooling_stride'] = self.pooling_stride[1:]
        inpt_dict_2d['is_2d'] = True

        self.torch_preprocessing = torch_preprocessing(**inpt_dict_3d)
        self.np_preprocessing = np_preprocessing(**inpt_dict_3d)

        self.torch_preprocessing_2d = torch_preprocessing(**inpt_dict_2d)
        self.np_preprocessing_2d = np_preprocessing(**inpt_dict_2d)

        self.is_initalised = True



    def is_preprocessed_data_tpl(self, data_tpl):
        return 'orig_shape' in data_tpl

    def __call__(self, data_tpl, preprocess_only_im=False, return_np=False):

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None
        if 'image' not in data_tpl:
            raise ValueError('No \'image\' found in data_tpl')

        xb = self.get_xb_from_data_tpl(data_tpl, preprocess_only_im)
        # now do the preprocessing
        if not torch.cuda.is_available():
            # the preprocessing is also faster in scipy then it is in torch using the CPU
            xb_prep = self.np_preprocessing(xb, spacing)[0]
        else:
            # when CUDA is available we will try to preprocess the data tuple on the GPU...
            xb_cuda = torch.from_numpy(xb).type(torch.float).cuda()
            try:
                xb_prep = self.torch_preprocessing(xb_cuda, spacing)[0]
                if return_np:
                    xb_prep = xb_prep.cpu().numpy()
            except RuntimeError:
                # ... unless it fails for a RuntimeError then we will try again on the CPU
                print('Ooops! It seems like your GPU has gone out of memory while trying to '
                      'resize a large volume ({}), trying again on the CPU.'
                      ''.format(list(xb_cuda.shape)))
                torch.cuda.empty_cache()
                xb_prep = self.np_preprocessing(xb, spacing)[0]
                if not return_np:
                    # if we don't want to return the numpy array we're brining it
                    # back to the GPU
                    xb_prep = torch.from_numpy(xb_prep).type(torch.float).cuda()

        return xb_prep

    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):

        # getting the image
        xb = data_tpl['image'].astype(float)

        if self.is_cascade():
            # the cascade is only implemented with binary predictions so far --> overwrite
            # this function for different predictions
            prev_preds = []
            for key in self.keys_for_previous_stages:
                assert key in data_tpl, 'prediction ' + key + ' from previous stage missing'
                pred = data_tpl[key]
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()
                # ensure the array is 4d
                prev_preds.append(pred)

            bin_pred = (np.sum(prev_preds, 0) > 0).astype(float)

            xb = np.concatenate([xb, bin_pred])

        # finally add batch axis
        xb = xb[np.newaxis]

        return xb

    def preprocess_raw_data(self,
                            folder,
                            save_as_fp16=True,
                            image_folder=None,
                            dcm_revers=True,
                            dcm_names_dict=None):


        print()
        raw_ds = raw_Dataset(join(folder, 'images'),
                             image_folder=image_folder,
                             dcm_revers=dcm_revers,
                             dcm_names_dict=dcm_names_dict,
                             prev_stages=self.prev_stages if self.is_cascade() else None,
                             create_missing_labels_as_zero=True,
                             additional_channels=self.num_additional_channels)

        if not self.is_initalised:
            print('Preprocessing classes were not initialised when calling '
                  '\'preprocess_raw_data\'. Doing it now.\n')
            self.initialise_preprocessing()

        im_dtype = np.float16 if save_as_fp16 else np.float32

        # root folder of all saved preprocessed data
        outfolder = join(folder, 'preprocessed_{}'.format(self.target_spacing[0]))
        if not exists(outfolder):
            mkdir(outfolder)

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        # here is the fun
        for i in tqdm(range(len(raw_ds))):
            # read files
            data_tpl = raw_ds[i]
            im, spacing = data_tpl['image'], data_tpl['spacing']
            scan = data_tpl['scan']

            orig_shape = im.shape[-3:]
            orig_spacing = spacing.copy()

            # get the preprocessed volumes from the data_tpl
            xb = self.__call__(data_tpl, return_np=True,)
            im = xb[:self.n_im_channels].astype(im_dtype)

            np.save(join(outfolder, scan), im)


        print('Preprocessing done!')



# %% Let's be fancy and do the preprocessing for np and torch as seperate operators
class torch_preprocessing(torch.nn.Module):

    # preprocessing module for 2d and 3d

    def __init__(self,
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=[1, 0],
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_2d = is_2d
        self.do_nn_img_interp = do_nn_img_interp

        # let's test if the inputs were fine
        if self.apply_resizing:
            self.target_spacing = np.array(target_spacing)
            if self.is_2d:
                assert len(target_spacing) == 2, 'target spacing must be of length 2'
                self.mode = 'bilinear'
            else:
                assert len(target_spacing) == 3, 'target spacing must be of length 3'
                self.mode = 'trilinear'

        if self.do_nn_img_interp:
            self.mode = 'nearest'

        if self.apply_pooling:
            if self.is_2d:
                assert len(pooling_stride) == 2, 'pooling stride must be of length 3'
                self.pooling_stride = pooling_stride
                self.mean_pooling = torch.nn.AvgPool2d(kernel_size=self.pooling_stride,
                                                       stride=self.pooling_stride)
                self.max_pooling = torch.nn.MaxPool2d(kernel_size=self.pooling_stride,
                                                      stride=self.pooling_stride)
            else:
                assert len(pooling_stride) == 3, 'pooling stride must be of length 3'
                self.pooling_stride = pooling_stride
                self.mean_pooling = torch.nn.AvgPool3d(kernel_size=self.pooling_stride,
                                                       stride=self.pooling_stride)
                self.max_pooling = torch.nn.MaxPool3d(kernel_size=self.pooling_stride,
                                                      stride=self.pooling_stride)

        if self.apply_windowing:
            assert len(window) == 2, 'window must be of length 2'
            self.window = window

        assert len(scaling) == 2, 'scaling must be of length 2 (std, mean)'
        self.scaling = scaling

    def forward(self, xb, spacing=None):

        # assume the image channels are always first
        n_ch = xb.shape[1]
        imb = xb[:, :self.n_im_channels]
        has_lb = n_ch > self.n_im_channels
        if has_lb:
            lbb = xb[:, self.n_im_channels:]

        # resizing
        if self.apply_resizing:

            scale_factor = (spacing / self.target_spacing).tolist()

            imb = interpolate(imb, scale_factor=scale_factor, mode=self.mode)
            if has_lb:
                lbb = interpolate(lbb, scale_factor=scale_factor)

        # pooling
        if self.apply_pooling:

            imb = self.mean_pooling(imb)
            if has_lb:
                lbb = self.max_pooling(lbb)

        # windowing:
        if self.apply_windowing:
            imb = imb.clamp(*self.window)

        # scaling
        imb = (imb - self.scaling[1]) / self.scaling[0]

        if has_lb:
            xb = torch.cat([imb, lbb], 1)
        else:
            xb = imb

        return xb


# %%
class np_preprocessing():

    # preprocessing class for 2d and 3d np niput

    def __init__(self,
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=[1, 0],
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels

        self.is_2d = is_2d
        self.do_nn_img_interp = do_nn_img_interp
        self.img_order = 0 if self.do_nn_img_interp else 1

        # let's test if the inputs were fine
        if self.apply_resizing:
            self.target_spacing = np.array(target_spacing)
            if self.is_2d:
                assert len(target_spacing) == 2, 'target spacing must be of length 2'
            else:
                assert len(target_spacing) == 3, 'target spacing must be of length 3'

        if self.apply_pooling:
            self.pooling_stride = pooling_stride
            if self.is_2d:
                assert len(pooling_stride) == 2, 'pooling stride must be of length 3'
            else:
                assert len(pooling_stride) == 3, 'pooling stride must be of length 3'

        if self.apply_windowing:
            assert len(window) == 2, 'window must be of length 2'
            self.window = window

        assert len(scaling) == 2, 'scaling must be of length 2 (std, mean)'
        self.scaling = scaling

    def _rescale_batch(self, im, spacing, order=1):

        if spacing is None:
            raise ValueError('spacing must be given as input when apply_resizing=True.')

        bs, nch = im.shape[0:2]
        idim = 2 if self.is_2d else 3
        scale = np.array(spacing) / self.target_spacing
        shape = im.shape[-1 * idim:]
        im_vec = im.reshape(-1, *shape)
        im_vec = np.stack([rescale(im_vec[i], scale, order=order) for i in range(im_vec.shape[0])])
        return im_vec.reshape(bs, nch, *im_vec.shape[1:])

    def __call__(self, xb, spacing=None):

        inpt_dim = 4 if self.is_2d else 5
        assert len(xb.shape) == inpt_dim, 'input images must be {}d tensor'.format(inpt_dim)

        # assume the image channels are always first
        n_ch = xb.shape[1]
        imb = xb[:, :self.n_im_channels]
        has_lb = n_ch > self.n_im_channels
        if has_lb:
            lbb = xb[:, self.n_im_channels:]

        # resizing
        if self.apply_resizing:

            imb = self._rescale_batch(imb, spacing, order=self.img_order)
            if has_lb:
                lbb = self._rescale_batch(lbb, spacing, order=0)

        # pooling
        if self.apply_pooling:

            imb = block_reduce(imb, (1, 1, *self.pooling_stride), func=np.mean)
            if has_lb:
                lbb = block_reduce(lbb, (1, 1, *self.pooling_stride), func=np.max)

        # windowing:
        if self.apply_windowing:
            imb = imb.clip(*self.window)

        # scaling
        imb = (imb - self.scaling[1]) / self.scaling[0]

        if has_lb:
            xb = np.concatenate([imb, lbb], 1)
        else:
            xb = imb

        return xb

if __name__ == '__main__':
    preprocess = SegmentationPreprocessing(apply_resizing=True,
                                             apply_pooling=False,
                                             apply_windowing=True,
                                             target_spacing=[4, 4, 4],
                                             window=[-50, 300],
                                             scaling=[60, 125],
                                             n_im_channels=1)

    preprocess.preprocess_raw_data('bask/projects/p/phwq4930-renal-canc/data/SSL_Data/CT')
    preprocess.preprocess_raw_data('bask/projects/p/phwq4930-renal-canc/data/SSL_Data/DeepLesion')