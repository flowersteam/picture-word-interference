import os
import urllib.request
import zipfile
import gzip

#if not os.path.exists("DATASETS/dataset1.mat"):
#    urllib.request.urlretrieve("http://wednesday.csail.mit.edu/MEG1_MEG_Clear_Data/visual_stimuli.mat",
#                               "DATASETS/dataset1.mat")

if not os.path.exists("DATASETS/dataset2.mat"):
    urllib.request.urlretrieve(
        "http://userpage.fu-berlin.de/rmcichy/Khaligh_Razavi_et_al_2018JoCN/118_visual_stimuli.mat",
        "DATASETS/dataset2.mat")

if not os.path.exists("DATASETS/dataset3.mat"):
    urllib.request.urlretrieve("http://wednesday.csail.mit.edu/fusion_rep/stimulus/156ImageStimuliSet.zip",
                               "DATASETS/dataset3.zip")
    with zipfile.ZipFile("DATASETS/dataset3.zip", 'r') as zip_file:
        with open("DATASETS/dataset3.mat", 'wb') as f:
            f.write(zip_file.read('156ImageStimuliSet/visual_stimuli156.mat'))

if not os.path.exists("DATASETS/GoogleNews-vectors-negative300.bin"):
    urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
                               "DATASETS/GoogleNews-vectors-negative300.bin.gz")
    with gzip.open("DATASETS/GoogleNews-vectors-negative300.bin.gz", 'r') as zip_ref:
        with open("DATASETS/GoogleNews-vectors-negative300.bin", 'wb') as f:
            f.write(zip_ref.read())