#pycerr_convert.py
# python pycerr_convert.py $CONVERT_OPT $DCM_PATH $OUTPUT_NAME $NII_MASK $JSON_SEGLIST

import os, sys, json

from cerr import plan_container as pc
from cerr.dcm_export import rtstruct_iod


def pycerr_dcm2nii(dcmDir,niiFileName = 'scan.nii.gz',scanNum=0):
    planC = pc.loadDcmDir(dcmDir = dcmDir)
    planC.scan[scanNum].saveNii(niiFileName = niiFileName)
    return niiFileName, planC


def pycerr_mask2rtstruct(dcmDir, maskFile, labels_dict = {1:'ROI'}, rtstructFileName = 'RTSTRUCT.dcm', seriesDescription = 'Imported by pyCERR', scanNum = 0, structNumV = None):
    planC = pc.loadDcmDir(dcmDir = dcmDir)
    if structNumV == None:
        structNumV = range(len(labels_dict))
    planC = pc.loadNiiStructure(nii_file_name = maskFile, assocScanNum = scanNum, planC = planC, labels_dict = labels_dict)
    rtstruct_iod.create(structNumV = structNumV, filePath = rtstructFileName, planC = planC, seriesOpts = {'SeriesDescription':seriesDescription})
    return rtstructFileName, planC


convert_opt = sys.argv[1]
dcm_path = sys.argv[2]
save_filename = sys.argv[3]

if convert_opt.lower() == 'dcm2nii':
    print('converting dicom folder ' + dcm_path + ' and outputting ' + save_filename)
    niiOut, planC = pycerr_dcm2nii(dcmDir=dcm_path, niiFileName = save_filename)
    print(niiOut)
elif convert_opt.lower() == 'mask2rtstruct':
    nii_mask = sys.argv[4]
    json_segfile = sys.argv[5]
    with open(json_segfile, 'r') as jfile:
        labels_dict = json.load(jfile)
    pycerr_mask2rtstruct(dcmDir = dcm_path, maskFile = nii_mask, labels_dict = labels_dict, rtstructFileName = save_filename, seriesDescription = 'Imported by pyCERR-CGC')

