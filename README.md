# GMFE
Generalized Multiscale Feature Extraction for Remaining Useful Life Prediction of Bearings with Generative Adversarial Networks

*Examples of execution

python3 train_step1_GAN.py --gpu=0 --condition=1 --test_bearing=2 --num_bearings=5 --n_channels=2 --lr=1e-4 --epoch=50 --mode=all --discriminator_lr=1e-4 --lambda_unet=50 --dropout=1 --batch_size=64 --dataset=xjtu

python3 train_step2_GAN.py --gpu=0 --condition=2 --test_bearing=2 --n_channels=3 --num_bearings=2 --lr=5e-5 --epoch=50 --mode=all --discriminator_lr=1e-4 --lambda_1=100 --lambda_2=50 --lambda_3=20 --dropout=1 --sequence_length=4 --batch_size=40 --dataset=xjtu --FPT=1 
