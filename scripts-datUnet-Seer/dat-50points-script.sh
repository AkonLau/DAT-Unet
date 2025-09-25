python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type DAT_UNet --batch_size 32  --epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type SwimUNet --batch_size 32  --epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type UNet --batch_size 32  --epochs 20

python test.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type DAT_UNet --batch_size 32  --save_epochs 20
python test.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type SwimUNet --batch_size 32  --save_epochs 20
python test.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type UNet --batch_size 32  --save_epochs 20

python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type TransUNet --batch_size 32  --epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type RadioUNet --batch_size 32  --epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type CBAMUNet --batch_size 32  --epochs 20

python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type TransUNet --batch_size 32  --save_epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type RadioUNet --batch_size 32  --save_epochs 20
python main.py --data_name sear --simulation DPM --image_size 256 --model_name unet --sp_ratio 10 --model_type CBAMUNet --batch_size 32  --save_epochs 20