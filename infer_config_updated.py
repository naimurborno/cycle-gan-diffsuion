# Inference Config
# 2D Cycle GAN configuration file

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######

config["data_path"] = "D:/Zakaria/3T to 7T MRI/content/T2 MRI CycleGAN/Data"  # Path for the input images.
# keep the image in either "Test" or "Val" subfolder
config["sub_fold"] = "Test"  # 'Test' or'Val'
config["model_name"] = "D:/Zakaria/3T to 7T MRI/content/T2 MRI CycleGAN/Results/resnetgen9_pixel_mae"  # folder where result is saved

# List of checkpoint filenames
config["ckpt_names"] = [
    "cyclegan-epoch=00004-step=9730.ckpt",
    "cyclegan-epoch=00009-step=19460.ckpt",
    "cyclegan-epoch=00014-step=29190.ckpt",
    "cyclegan-epoch=00019-step=38920.ckpt",
    "cyclegan-epoch=00024-step=48650.ckpt",
    "cyclegan-epoch=00029-step=58380.ckpt",
    "cyclegan-epoch=00034-step=68110.ckpt",
    "cyclegan-epoch=00039-step=77840.ckpt",
    "cyclegan-epoch=00044-step=87570.ckpt",
    "cyclegan-epoch=00049-step=97300.ckpt",
    "cyclegan-epoch=00054-step=107030.ckpt",
    "cyclegan-epoch=00059-step=116760.ckpt",
    "cyclegan-epoch=00064-step=126490.ckpt",
    "cyclegan-epoch=00069-step=136220.ckpt",
    "cyclegan-epoch=00074-step=145950.ckpt",
    "cyclegan-epoch=00079-step=155680.ckpt",
    "cyclegan-epoch=00084-step=165410.ckpt",
    "cyclegan-epoch=00089-step=175140.ckpt",
    "cyclegan-epoch=00094-step=184870.ckpt",
    "cyclegan-epoch=00099-step=194600.ckpt",
    "cyclegan-epoch=00104-step=204330.ckpt",
    "cyclegan-epoch=00109-step=214060.ckpt",
    "cyclegan-epoch=00114-step=223790.ckpt",
    "cyclegan-epoch=00119-step=233520.ckpt",
    "cyclegan-epoch=00124-step=243250.ckpt",
    "cyclegan-epoch=00129-step=252980.ckpt",
    "cyclegan-epoch=00134-step=262710.ckpt",
    "cyclegan-epoch=00139-step=272440.ckpt",
    "cyclegan-epoch=00144-step=282170.ckpt",
    "cyclegan-epoch=00149-step=291900.ckpt",
    "cyclegan-epoch=00154-step=301630.ckpt",
    "cyclegan-epoch=00159-step=311360.ckpt",
    "cyclegan-epoch=00164-step=321090.ckpt",
    "cyclegan-epoch=00169-step=330820.ckpt",
    "cyclegan-epoch=00174-step=340550.ckpt",
    "cyclegan-epoch=00179-step=350280.ckpt",
    "cyclegan-epoch=00184-step=360010.ckpt",
    "cyclegan-epoch=00189-step=369740.ckpt",
    "cyclegan-epoch=00194-step=379470.ckpt",
    "cyclegan-epoch=00199-step=389200.ckpt"
]

config["paired"] = False  # For Aligned task set to True. Otherwise False
config["batch_size"] = 1  # batch size. keep it to 1 for least bugs
config["gen_model"] = "resnet_gen_9"  # ['resnet_gen_9','resnet_gen_7','resnet_gen_3']
config["dis_model"] = "pixelGAN"  # ['patchGAN','pixelGAN']

### Metrics to compute
# config["metrics"] = ["fid","psnr","ssim"]  # ['ssim', 'fid', 'psnr']. set it to [] for no cal
