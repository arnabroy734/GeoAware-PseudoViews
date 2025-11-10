#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
from pathlib import Path
import sys
import numpy as np

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--dest_path", "-d", required=True, type=str)
parser.add_argument("--camera", default="PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--n_views", default=0, type=int)
parser.add_argument("--stereo_fusion", action='store_true')
parser.add_argument("--train_test_split", action='store_true')



args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

if not args.skip_matching:
    # The source folder should contain all the images, split will be created by the code dynamically
    # - if not put n_views in train if n_views > 0 else put every 8th in test and rest in train
    base_source_path = Path(args.source_path).expanduser()  
    base_source_path = base_source_path.resolve()
    trainpath = base_source_path/'train'
    testpath = base_source_path/'test'
    if args.train_test_split:
        if not trainpath.exists() or not testpath.exists(): 
            all_images = [file for file in base_source_path.iterdir() if not file.is_dir() and file.suffix in ['.png', '.jpg']]
            np.random.shuffle(all_images)
            Path.mkdir(trainpath)
            Path.mkdir(testpath)
            if args.n_views > 0: 
                train_idx = [round_python3(i) for i in np.linspace(0, len(all_images)-1, args.n_views)]
            else: 
                train_idx = [i for i,_ in enumerate(all_images) if i%8 != 0] 
            for idx, imagefile in enumerate(all_images): 
                if idx in train_idx: 
                    shutil.copy(imagefile, trainpath/imagefile.name)
                else: 
                    shutil.copy(imagefile, testpath/imagefile.name)
        else: 
            raise Exception('Train and test path already exists ...')
    
    dest_root = args.dest_path
    args.dest_path = args.dest_path + '/train'
    os.makedirs(args.dest_path + "/distorted/sparse", exist_ok=True)
    
    if not trainpath.exists():
        print('Train folder does not exist')
        sys.exit(1)
    if not testpath.exists():
        print('Test path does not exist')
        sys.exit(1)
    

    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.dest_path + "/distorted/database.db \
        --image_path " + args.source_path + "/train \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) + """ \
        --SiftExtraction.max_image_size 4032 \
        --SiftExtraction.max_num_features 32768 \
        --SiftExtraction.estimate_affine_shape 1 \
        --SiftExtraction.domain_size_pooling 1 \
        """ 
    
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.dest_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu) + " --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768"
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.dest_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/train \
        --output_path "  + args.dest_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/train \
    --input_path " + args.dest_path + "/distorted/sparse/0 \
    --output_path " + args.dest_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.dest_path + "/sparse")
os.makedirs(args.dest_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.dest_path, "sparse", file)
    destination_file = os.path.join(args.dest_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.dest_path + "/images_2", exist_ok=True)
    os.makedirs(args.dest_path + "/images_4", exist_ok=True)
    os.makedirs(args.dest_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.dest_path, "images", file)

        destination_file = os.path.join(args.dest_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.dest_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.dest_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("--Training Images SFM Done-- \n")

# Stereo Fusion train images
if args.stereo_fusion: 
    # copy all files from spares/0 to sparse
    destination_path = Path(args.dest_path).expanduser()  
    destination_path = destination_path.resolve()
    for file in (destination_path/'sparse/0').iterdir(): 
        shutil.copy(file, destination_path/f'sparse/{file.name}')
    exitcode = os.system('colmap patch_match_stereo --workspace_path '+ f'{str(destination_path)}')
    if exitcode != 0: 
        print(f'patch match_stereo failed {exit_code}')
        exit(exitcode)       
    exitcode = os.system('colmap stereo_fusion --workspace_path '+ f'{str(destination_path)}' + ' --output_path '+ f'{str(destination_path)}'+'/fused.ply')
    if exitcode != 0: 
        print(f'Stereo fusion failed {exit_code}')
        exit(exitcode) 
    print("----STEREO FUSION DONE ------\n")

print("-"*30)
print('Registering Test Images')

# feature extraction
feat_extracton_cmd = colmap_command + " feature_extractor "\
    "--database_path " + args.dest_path + "/distorted/database.db \
    --image_path " + args.source_path + "/test \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model " + args.camera + " \
    --SiftExtraction.use_gpu " + str(use_gpu) + """ \
    --SiftExtraction.max_image_size 4032 \
    --SiftExtraction.max_num_features 32768 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1 \
    """
exit_code = os.system(feat_extracton_cmd)
if exit_code != 0:
    logging.error(f"Feature extraction failed with code {exit_code}. FOR TEST IMAGES.")
    exit(exit_code)

# feature matching
feat_matching_cmd = colmap_command + " exhaustive_matcher \
    --database_path " + args.dest_path + "/distorted/database.db \
    --SiftMatching.use_gpu " + str(use_gpu) +" --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768"
exit_code = os.system(feat_matching_cmd)
if exit_code != 0:
    logging.error(f"Feature matching failed with code {exit_code}. FOR TEST IMAGES.")
    exit(exit_code)

# Image registration
os.makedirs(dest_root + "/test", exist_ok=True)
os.makedirs(dest_root + "/test/sparse/0", exist_ok=True)
os.makedirs(dest_root + "/test/images", exist_ok=True)

image_registration_cmd = colmap_command + " image_registrator \
    --database_path " + args.dest_path + "/distorted/database.db \
    --input_path " + args.dest_path + "/sparse/0 \
    --output_path " + dest_root + "/test/sparse/0"
exit_code = os.system(image_registration_cmd)
if exit_code != 0:
    logging.error(f"TEST images registration failed {exit_code}. FOR TEST IMAGES.")
    exit(exit_code)

# copy all test images to folder
for file in testpath.iterdir():
    shutil.copy(file, Path.cwd()/dest_root/'test/images')

if args.train_test_split:
    # In case new train test folder is created, then remove the created folders
    shutil.rmtree(trainpath)
    shutil.rmtree(testpath)

# convert model
os.system('colmap model_converter  --input_path '+ args.dest_path+'/sparse/0 '+ '--output_path ' + args.dest_path+'/sparse/0' + ' --output_type TXT')
os.system('colmap model_converter  --input_path '+ dest_root+'/test/sparse/0 '+ '--output_path ' + dest_root+'/test/sparse/0' + ' --output_type TXT')
print('PIPELINE COMPLETE')
