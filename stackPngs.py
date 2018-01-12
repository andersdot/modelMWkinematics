from fpdf import FPDF
import glob
import numpy as np

filename_search = 'test_dist*_lpole*_bpole*_phi1*.png'
allimages = np.array(glob.glob(filename_search))

#want to bring together images with the same pole
#test_dist020_lpole99.64_bpole48.14_phi1090_110.png
poles = [image.split('_')[2] + '_' + image.split('_')[3] for image in allimages]
uniqpoles = np.unique(poles)

for pole in uniqpoles:
    poleind = np.in1d(poles, pole)
    list_of_images = allimages[poleind]
    w = 200
    h = 200
    pdf = FPDF()
    for image in list_of_images:
        pdf.add_page()
        #pdf.cell(30, 10, image, 1, 0, 'C')
        pdf.image(image, w=w, h=h)
    pdf.output(pole + '.pdf', 'F')
