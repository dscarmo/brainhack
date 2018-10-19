import pickle
import numpy as np
import cv2 as cv
from train import *
from rdataset import *
from skimage import measure as m

import torch
import torch.nn as nn 

plt.ion()

model_name = "compressed_sensing_test_nolinear_50epochs"
with open(str(model_name) + ".pkl", "rb") as output_file:
    results = pickle.load(output_file)


rec_dataloaders, device, dataset_sizes = prepare_environment(batch_size=1) 

results.plot()

model = results.best_model

rec_it = iter(rec_dataloaders["test"])

mse = nn.MSELoss()
l1 = nn.L1Loss()

mses = []
l1s = []
nrmses = []
pre_nrmses = []

i = 0
for uimg, img in rec_it:
    gpu_uimg = uimg.to(device)
    output = model(gpu_uimg)

    mses.append(mse(output, gpu_uimg).item())
    l1s.append(l1(output, gpu_uimg).item())
    
    # Convert to numpy
    output = output.cpu().detach().squeeze().numpy()
    uimg_display = uimg.squeeze().numpy()
    img_display = img.squeeze().numpy()
    
    nrmses.append(m.compare_nrmse(img_display, output, norm_type='min-max'))
    pre_nrmses.append(m.compare_nrmse(uimg_display, output, norm_type='min-max'))

    # Normalize
    #cv.normalize(output, output, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    #cv.normalize(uimg_display, uimg_display, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    #cv.normalize(img_display, img_display, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    
    #output = np.ma.masked_less(output, 0.15).filled(fill_value=0.0)

    before_diff = np.abs(uimg_display - img_display) 
    after_diff = np.abs(output - img_display)
    
    cv.putText(img_display, "Target image", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(img_display, "From full k-space", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv.putText(uimg_display, "Input image", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(uimg_display, "From undersampled k-space", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv.putText(output, "Predicted image", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(after_diff, "|predicted-target|", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(before_diff, "|input-target|", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    out_display = np.hstack([uimg_display, img_display, output, after_diff, before_diff])  
    
    #cv.imshow("Input - Output - Target - Differences", out_display)
    cv.imshow("Upsampled Input - Output - Target - Differences", cv.resize(out_display, (1920, 900), interpolation=cv.INTER_CUBIC))
    
    '''pred = (output - output.min())/output.max()

    cv.imwrite("pics\\" + str(i) + "prediction.png", (pred*255).astype(np.uint8))
    cv.imwrite("pics\\" + str(i) + "input.png", (uimg_display*255).astype(np.uint8))
    cv.imwrite("pics\\" + str(i) + "target.png", (img_display*255).astype(np.uint8))
    cv.imwrite("pics\\" + str(i) + "predicted-target.png", (after_diff*255).astype(np.uint8))
    cv.imwrite("pics\\" + str(i) + "input-target.png", (before_diff*255).astype(np.uint8))'''
    
    if cv.waitKey(1) == 27:
        break
    i += 1

print()
print("Pre-network mean NRMSE: {}".format(np.sum(np.array(pre_nrmses))/len(pre_nrmses)))

print("After network Test Mean Square Error: {}\nTest absolute error: {}\nTest normalized mean Root Mean-Squared Error (NRMSE): {} +- {}".format(np.sum(np.array(mses))/len(mses), np.sum(np.array(l1s))/len(l1s), np.sum(np.array(nrmses))/len(nrmses), np.array(nrmses).std()))