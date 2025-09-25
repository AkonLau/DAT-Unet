# python main.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type UNet --batch_size 32  --epochs 100  --in_channels 3
# python main.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type CBAMUNet --batch_size 32  --epochs 100  --in_channels 3
# python main.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type SwimUNet --batch_size 32  --epochs 100  --in_channels 3
# python main.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type DAT_UNet-t2 --batch_size 32  --epochs 100  --in_channels 3

python test.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type UNet --batch_size 32  --epochs 100  --in_channels 3
python test.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type CBAMUNet --batch_size 32  --epochs 100  --in_channels 3
python test.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type SwimUNet --batch_size 32  --epochs 100  --in_channels 3
python test.py --data_name pp5d-h0 --image_size 128 --seed 1 --model_name unet --sp_ratio 10 --model_type DAT_UNet-t2 --batch_size 32  --epochs 100  --in_channels 3
