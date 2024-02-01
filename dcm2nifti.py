import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files

import compressed_dicom

import gc
import os
import re
import traceback
import unicodedata

from pydicom.tag import Tag

import logging

import dicom2nifti.common as common
import dicom2nifti.convert_dicom as convert_dicom
import dicom2nifti.settings
import torch
logger = logging.getLogger(__name__)


def convert_directory(dicom_directory, output_filename, compression=True, reorient=True):
    """
    This function will order all dicom files by series and order them one by one

    :param compression: enable or disable gzip compression
    :param reorient: reorient the dicoms according to LAS orientation
    :param output_filename: filename to write the nifti files to
    :param dicom_directory: directory with dicom files
    """
    # sort dicom files by series uid
    dicom_series = {}
    for root, _, files in os.walk(dicom_directory):
        for dicom_file in files:
            file_path = os.path.join(root, dicom_file)
            # noinspection PyBroadException
            try:
                if compressed_dicom.is_dicom_file(file_path):
                    # read the dicom as fast as possible
                    # (max length for SeriesInstanceUID is 64 so defer_size 100 should be ok)

                    dicom_headers = compressed_dicom.read_file(file_path,
                                                               defer_size="1 KB",
                                                               stop_before_pixels=False,
                                                               force=dicom2nifti.settings.pydicom_read_force)
                    if not _is_valid_imaging_dicom(dicom_headers):
                        logger.info("Skipping: %s" % file_path)
                        continue
                    logger.info("Organizing: %s" % file_path)
                    if dicom_headers.SeriesInstanceUID not in dicom_series:
                        dicom_series[dicom_headers.SeriesInstanceUID] = []
                    dicom_series[dicom_headers.SeriesInstanceUID].append(dicom_headers)
            except:  # Explicitly capturing all errors here to be able to continue processing all the rest
                logger.warning("Unable to read: %s" % file_path)
                traceback.print_exc()

    # start converting one by one
    for series_id, dicom_input in dicom_series.items():
        base_filename = ""
        # noinspection PyBroadException
        try:
            # construct the filename for the nifti
            base_filename = ""
            if 'SeriesNumber' in dicom_input[0]:
                base_filename = _remove_accents('%s' % dicom_input[0].SeriesNumber)
                if 'SeriesDescription' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].SeriesDescription))
                elif 'SequenceName' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].SequenceName))
                elif 'ProtocolName' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].ProtocolName))
            else:
                base_filename = _remove_accents(dicom_input[0].SeriesInstanceUID)
            logger.info('--------------------------------------------')
            logger.info('Start converting %s' % base_filename)
            convert_dicom.dicom_array_to_nifti(dicom_input, output_filename, reorient)
            gc.collect()
        except:  # Explicitly capturing app exceptions here to be able to continue processing
            logger.info("Unable to convert: %s" % base_filename)
            traceback.print_exc()



def _is_valid_imaging_dicom(dicom_header):
    """
    Function will do some basic checks to see if this is a valid imaging dicom
    """
    # if it is philips and multiframe dicom then we assume it is ok
    try:
        if common.is_philips([dicom_header]) or common.is_siemens([dicom_header]):
            if common.is_multiframe_dicom([dicom_header]):
                return True

        if "SeriesInstanceUID" not in dicom_header:
            return False

        if "InstanceNumber" not in dicom_header:
            return False

        if "ImageOrientationPatient" not in dicom_header or len(dicom_header.ImageOrientationPatient) < 6:
            return False

        if "ImagePositionPatient" not in dicom_header or len(dicom_header.ImagePositionPatient) < 3:
            return False

        # for all others if there is image position patient we assume it is ok
        if Tag(0x0020, 0x0037) not in dicom_header:
            return False

        return True
    except (KeyError, AttributeError):
        return False


def _remove_accents(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    # noinspection PyBroadException
    try:
        unicode_filename = unicode_filename.replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')

        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)

        return cleaned_filename
    except:
        traceback.print_exc()
        return unicode_filename


def _remove_accents_(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    valid_characters = bytes(b'-_.() 1234567890abcdefghijklmnopqrstuvwxyz')
    cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore')

    new_filename = ""

    for char_int in bytes(cleaned_filename):
        char_byte = bytes([char_int])
        if char_byte in valid_characters:
            new_filename += char_byte.decode()

    return new_filename


if __name__ == '__main__':
    MM_home = '/bask/projects/p/phwq4930-renal-canc/data/SSL_Data/'
    MR_dir = os.path.join(MM_home, 'MR')
    CT_dir = os.path.join(MM_home, 'CT')

    for mode_path in [CT_dir,MR_dir]:
        m_im_p = os.path.join(mode_path, 'images')
        if not os.path.exists(m_im_p): os.makedirs(m_im_p)

        for dataset in [dir for dir in os.listdir(mode_path) if (os.path.isdir(os.path.join(mode_path,dir))) and (dir!='images')]:
            for series in [dir for dir in os.listdir(os.path.join(mode_path, dataset)) if (os.path.isdir(os.path.join(mode_path,dataset,dir)))]:
                s_im_p = os.path.join(m_im_p, dataset+'#'+series+'.nii.gz')
                try:
                    convert_directory(os.path.join(mode_path, dataset, series), s_im_p, compression=True,
                                              reorient=True)
                except:
                    print('Failed to convert directory')
                    continue

                nibim = nib.load(s_im_p)
                im = nibim.get_fdata()
                #if 4d, take first volume. find the channel-wise dim using argmin
                if len(im.shape) == 4:
                    print('4d image',dataset,series)
                    im = torch.Tensor(im).select(torch.argmin(torch.Tensor(list(im.shape))), 0)
                    affine = nibim.affine
                    # check affine is 4x4
                    assert affine.shape == (4, 4)
                    new_nibim = nib.Nifti1Image(im.numpy(), affine)
                    nib.save(new_nibim, s_im_p)
                elif len(im.shape) == 3:
                    continue
                else:
                    raise ValueError('Image shape is not 3d or 4d')