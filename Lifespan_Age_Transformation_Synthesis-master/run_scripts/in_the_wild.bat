@echo off

set CUDA_VISIBLE_DEVICES=0

python D:\github_project\Lifespan_Age_Transformation_Synthesis-master\test.py --name females_model --which_epoch latest --display_id 0 --traverse --interp_step 0.05 --image_path_file D:\github_project\Lifespan_Age_Transformation_Synthesis-master\run_scripts\path.txt --make_video --in_the_wild --verbose
pause