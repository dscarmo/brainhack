import pickle
import cv2 as cv
from train import *
from rdataset import *

model_name = "compressed_sensing_test_nolinear"
with open(str(model_name) + ".pkl", "rb") as output_file:
    results = pickle.load(output_file)

results.plot()

rec_dataloaders, device, dataset_sizes = prepare_environment(batch_size=1) 

model = results.best_model

rec_it = iter(rec_dataloaders["test"])

for uimg, img in rec_it:
    output = model(uimg.to(device))
    output = output.cpu()
    #out_display = (output - output.min())/output.max()
    out_display = output
    cv.imshow("Input", uimg.squeeze().numpy())
    cv.imshow("Output", out_display.squeeze().detach().numpy())
    if cv.waitKey(0) == 27:
        break
