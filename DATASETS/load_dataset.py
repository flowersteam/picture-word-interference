import scipy.io

# i : LABEL - HUMAN/NONHUMAN - FACE/BODY - ANIMATE/INANIMATE - NATURAL/ARTIFICIAL - PIXELS
# s_92 = scipy.io.loadmat('DATASETS/dataset1.mat')
# i : LABEL - PIXELS - ANIMATE - SMALL - MEDIUM - LARGE
s_118 = scipy.io.loadmat('DATASETS/dataset2.mat')
# i : PIXELS - TWINSET - ('animals', 'objects', 'scenes', 'people', or 'faces')
s_156 = scipy.io.loadmat('DATASETS/dataset3.mat')


def get_images():
    images = []
    for i in range(len(s_118['visual_stimuli'][0])):
        images.append(s_118['visual_stimuli'][0][i][1])
    for i in range(len(s_156['visual_stimuli156'][0])):
        images.append(s_156['visual_stimuli156'][0][i][0])
    return images
