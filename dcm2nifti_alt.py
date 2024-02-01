import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files


import os




if __name__ == '__main__':
    MM_home = '/Users/mcgoug01/Downloads/SSL_Data'
    MR_dir = os.path.join(MM_home, 'MR')
    CT_dir = os.path.join(MM_home, 'CT')

    for mode_path in [MR_dir,CT_dir]:
        m_im_p = os.path.join(mode_path, 'images_alt')
        if not os.path.exists(m_im_p): os.makedirs(m_im_p)

        for dataset in [dir for dir in os.listdir(mode_path) if (os.path.isdir(os.path.join(mode_path,dir))) and (dir!='images')]:
            for series in [dir for dir in os.listdir(os.path.join(mode_path, dataset)) if (os.path.isdir(os.path.join(mode_path,dataset,dir)))]:
                s_im_p = os.path.join(m_im_p, dataset+'#'+series+'.nii.gz')
                try:
                    dicom2nifti.convert_dicom.dicom_series_to_nifti(os.path.join(mode_path, dataset, series), s_im_p)
                except:
                    print('Failed to convert: '+s_im_p)
                    continue