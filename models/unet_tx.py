import os
import json
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.dat_models.datUnet import DAT_UNET
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class UNet_Tx(nn.Module):
    def __init__(self,
                 model_name,
                 model_type='UNet',
                 in_channels=3,
                 # map sample params
                 img_size=128,
                 patch_size=1,
                 sample_chans=1,
                 # UNet params
                 latent_channels=64,
                 out_channels=1,
                 features=[32, 32, 32]):

        self.config = dict(model_name=model_name,
                           model_type=model_type,
                           in_channels=in_channels,
                           img_size=img_size,
                           patch_size=patch_size,
                           sample_chans=sample_chans,
                           latent_channels=latent_channels,
                           out_channels=out_channels,
                           features=features)

        super(UNet_Tx, self).__init__()
        self.in_channels = in_channels


        if model_type == 'DAT_UNet':
            self.model2 = DAT_UNET(
                img_size=img_size,
                patch_size=4,
                in_chans=in_channels,
                out_chans=1,
                expansion=4,
                dim_stem=64,
                dims=[64, 128, 256, 512],
                depths=[4, 4, 4, 4],
                stage_spec=[
                    ['N', 'D', 'N', 'D'],
                    ['N', 'D', 'N', 'D'],
                    ['N', 'D', 'N', 'D'],
                    ['D', 'D', 'D', 'D']
                ],
                heads=[2, 4, 8, 16],
                window_sizes=[7, 7, 7, 7],
                groups=[1, 2, 4, 8],
                use_pes=[True, True, True, True],
                dwc_pes=[True, True, True, True],
                strides=[8, 4, 2, 1],
                offset_range_factor=[-1, -1, -1, -1],
                no_offs=[False, False, False, False],
                fixed_pes=[False, False, False, False],
                use_dwc_mlps=[True, True, True, True],
                use_lpus=[True, True, True, True],
                use_conv_patches=True,
                ksizes=[9, 7, 5, 3],
                nat_ksizes=[7, 7, 7, 7],
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2
            )

        else:
            print(f'{model_type} is not supported!')
            raise ValueError

    def forward(self, tx_map, building_mask, image_cars):
        # building_mask has 1 for free space, 0 for buildings (may change this for future datasets).
        inv_building_mask = 1 - building_mask

        # sample_mask has 1 for non-sampled locations, 0 for sampled locations.

        if self.in_channels == 3 and image_cars is not None:
            # print('image_cars is not None')
            image_cars = image_cars.to(torch.float32).to(device)

            x = torch.cat((tx_map, inv_building_mask, image_cars), dim=1)

        elif self.in_channels == 2:
            x = torch.cat((tx_map, inv_building_mask), dim=1)
        else:
            raise ValueError("Don't support input channel {}".format(self.in_channels))

        map2 = self.model2(x)

        return tx_map, inv_building_mask, map2

    def step(self, args, batch, optimizer, min_samples, max_samples, train=True, free_space_only=False):
        with torch.set_grad_enabled(train):
            if len(batch) == 4:
                complete_map, building_mask, image_cars, tx_map = batch
            elif len(batch) == 3:
                complete_map, building_mask, tx_map = batch
                image_cars = None
            elif len(batch) == 2:
                complete_map, building_mask = batch
                image_cars = None
            complete_map, building_mask = complete_map.to(torch.float32).to(device), building_mask.to(torch.float32).to(
                device)

            tx_map = tx_map.to(torch.float32).to(device)

            map1, _, pred_map = self.forward(tx_map, building_mask, image_cars)
            map1, pred_map = map1.to(torch.float32), pred_map.to(torch.float32)

            # building_mask has 1 for free space, 0 for buildings (may change this for future datasets)
            # RadioUNet also calculates loss over buildings, whereas our previous models did not. I have included both options here.
            if free_space_only:
                loss_ = nn.functional.mse_loss(pred_map * building_mask, complete_map * building_mask).to(torch.float32)
            else:
                loss_ = nn.functional.mse_loss(pred_map, complete_map).to(torch.float32)

            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_

    def fit_wandb(self, train_dl, val_dl, optimizer, scheduler, min_samples, max_samples, project_name, run_name=None,
                  dB_max=-47.84, dB_min=-147, free_space_only=False, epochs=100, save_model_epochs=25,
                  save_model_dir='/content', **kwargs):

        args = kwargs['args']
        if run_name is None:
            run_name = f"tx-only, {self.config['model_type']}"
            if args.simulation != 'DPM' and args.data_name == 'sear':
                run_name += f', {args.simulation}'
            if free_space_only:
                run_name += ', free space only'
            if args.gradient_loss:
                run_name += f', gradient_loss-{args.lambda_gradient}'
            if args.variation_loss:
                run_name += f', variation_loss-{args.lambda_variation}'

            if args.in_channels != 3:
                run_name += f", {self.in_channels}-in_channels"
            else:
                run_name += f", {self.in_channels}-in_channels, cars"
        print(run_name)

        self.training_config = dict(train_batch=train_dl.batch_size, val_batch=val_dl.batch_size,
                                    min_samples=min_samples,
                                    max_samples=max_samples, project_name=project_name, run_name=run_name,
                                    dB_max=dB_max,
                                    dB_min=dB_min, free_space_only=free_space_only)
        if args.wandb:
            import wandb
            config = {**self.config, **self.training_config, **vars(kwargs['args'])}
            wandb.init(project=project_name, group=config['model_name'], name=run_name, config=config)

        for epoch in range(epochs):
            self.train()
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(args, batch, optimizer, min_samples, max_samples, train=True,
                                 free_space_only=free_space_only)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss / (i + 1)

                if i % 20 == 0:
                    print(f'loss: {loss}, [{epoch + 1}, {i + 1:5d}] train_loss: {train_loss}')

            from util.tools import SeedContextManager
            with SeedContextManager(seed=args.seed):
                rmse_values, ssim_values, psnr_values = self.evaluate(val_dl, min_samples, max_samples, dB_max, dB_min,
                                                                      free_space_only=free_space_only)
            print(f'{rmse_values}, [{epoch + 1}]')

            if args.wandb:
                wandb.log({'train_loss': train_loss,
                           'test_rmse': rmse_values,
                           'test_ssim': ssim_values,
                           'test_psnr': psnr_values,
                           'train_lr': scheduler.get_last_lr()[0]
                           })

            if scheduler:
                scheduler.step()

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                self.save_model(epoch + 1, optimizer, scheduler, out_dir=save_model_dir)

    def evaluate(self, test_dl, min_samples, max_samples, dB_max=-47.84, dB_min=-147, free_space_only=True,
                 pre_sampled=False, **kwargs):
        ### free_space_only set as True for evaluating the area outside the building

        self.eval()
        losses = 0
        pixels = 0

        from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_values = 0
        psnr_values = 0
        nmse_values = 0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                if len(batch) == 4:
                    complete_map, building_mask, image_cars, tx_map = batch
                elif len(batch) == 3:
                    complete_map, building_mask, tx_map = batch
                    image_cars = None
                elif len(batch) == 2:
                    complete_map, building_mask = batch
                    image_cars = None

                building_mask = building_mask.to(torch.float32).to(device)
                tx_map = tx_map.to(torch.float32).to(device)

                complete_map = complete_map.to(torch.float32).to(device)
                _, _, pred_map = self.forward(tx_map, building_mask, image_cars)

                pred_map = pred_map.detach()

                ###visualization start###
                if 'args' in kwargs:
                    args = kwargs['args']
                    if args.visualization and i < 10:
                        visual_results = self.visualize_maps(complete_map, building_mask, tx_map, building_mask,
                                                             pred_map)

                        base_path = './saved_figures'
                        if 'run_dir' in kwargs:
                            run_dir = kwargs['run_dir']
                        save_path = os.path.join(base_path, args.model_name.lower(), run_dir)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        img_path = os.path.join(save_path, f"{i}.png")
                        # print(save_path)
                        visual_results[0].savefig(img_path)
                        plt.close()

                        plt.imshow(visual_results[1][0, 0, :, :].cpu())
                        plt.axis('off')
                        plt.savefig(os.path.join(save_path, f"input_image_{i}.png"), bbox_inches='tight', pad_inches=0)
                        plt.close()

                        from scipy.ndimage import zoom
                        scale_factor = 0.5
                        sample_map_np = visual_results[2][0, 0, :, :].cpu().numpy()
                        sample_map_rescaled = zoom(sample_map_np, scale_factor, order=1)

                        plt.imshow(sample_map_rescaled)
                        plt.axis('off')
                        plt.savefig(os.path.join(save_path, f"sample_image_{i}.png"), bbox_inches='tight', pad_inches=0)
                        plt.close()

                        sample_mask = building_mask
                        building_sample_map_np = sample_mask[0, 0, :, :].cpu()
                        building_sample_map_rescaled = zoom(building_sample_map_np, scale_factor, order=1)

                        plt.imshow(building_sample_map_rescaled)
                        plt.axis('off')
                        plt.savefig(os.path.join(save_path, f"building+sample_image_{i}.png"), bbox_inches='tight',
                                    pad_inches=0)
                        plt.close()

                        plt.imshow(visual_results[4][0, 0, :, :].cpu())
                        plt.axis('off')
                        plt.savefig(os.path.join(save_path, f"pred_image_{i}.png"), bbox_inches='tight', pad_inches=0)
                        plt.close()
                        # import pdb; pdb.set_trace()
                    # else:
                    #     exit(0)
                ###visualization end###

                complete_map = complete_map.to(torch.float32)
                building_mask = building_mask

                # building_mask has 1 for free space, 0 for buildings (may change this for future datasets)
                # RadioUNet also calculates loss over buildings, whereas our previous models did not. I have included both options here.
                if free_space_only:
                    loss = nn.functional.mse_loss(self.scale_to_dB(pred_map * building_mask, dB_max, dB_min),
                                                  self.scale_to_dB(complete_map * building_mask, dB_max, dB_min),
                                                  reduction='sum')
                    pix = building_mask.sum()
                    ssim_value = ssim_metric(pred_map * building_mask, complete_map * building_mask)
                    psnr_value = psnr_metric(pred_map * building_mask, complete_map * building_mask)
                    nmse_value = self.compute_nmse(pred_map * building_mask, complete_map * building_mask)

                else:
                    loss = nn.functional.mse_loss(self.scale_to_dB(pred_map, dB_max, dB_min),
                                                  self.scale_to_dB(complete_map, dB_max, dB_min),
                                                  reduction='sum')
                    pix = pred_map.numel()
                    ssim_value = ssim_metric(pred_map, complete_map)
                    psnr_value = psnr_metric(pred_map, complete_map)
                    nmse_value = self.compute_nmse(pred_map, complete_map)

                losses += loss
                pixels += pix
                # print(f'{torch.sqrt(loss / pix).item()}')

                batch_size = pred_map.size(0)
                ssim_values += ssim_value * batch_size
                psnr_values += psnr_value * batch_size
                nmse_values += nmse_value * batch_size
                total_samples += batch_size

            print(f"SSIM: {ssim_values / total_samples}")
            print(f"PSNR: {psnr_values / total_samples}")
            print(f"NMSE: {nmse_values / total_samples}")

            return math.sqrt(losses / pixels), ssim_values / total_samples, psnr_values / total_samples

    def visualize_maps(self, x, building_mask, sample_map, sample_mask, pred_map):
        pred_map *= building_mask
        pred_map + (1 - building_mask)

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(x[0, 0, :, :].cpu())
        axs[1].imshow(sample_mask[0, 0, :, :].detach().cpu())
        axs[2].imshow(sample_map[0, 0, :, :].detach().cpu())
        axs[3].imshow(pred_map[0, 0, :, :].detach().cpu())
        # import pdb; pdb.set_trace()
        # plt.show()
        return fig, x.cpu(), sample_mask.detach().cpu(), sample_map.detach().cpu(), pred_map.detach().cpu()

    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return torch.Tensor(dB)

    def save_model(self, epoch=0, optimizer=None, scheduler=None, out_dir='/content'):
        # First time model is saved (as indicated by not having a pre-existing model directory),
        # create model folder and save model config.
        model_name = self.config['model_name']
        model_dir = os.path.join(out_dir, model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

            # Save model config (i.e. params called with __init__).
            config_path = os.path.join(model_dir, f'{model_name} config.json')
            with open(config_path, 'w') as config_file:
                json.dump(self.config, config_file, indent=2)

        # If called with fit or fit_wandb (hence having "self.training_config"), make a new run
        # directory and save training_config, optimizer, scheduler, and trained weights.
        if hasattr(self, 'training_config'):
            run_name = self.training_config['run_name']
            run_dir = os.path.join(model_dir, run_name)
            if not os.path.isdir(run_dir):
                os.mkdir(run_dir)

                # Save training_config first time for new run
                train_config_path = os.path.join(run_dir, 'training_config.json')
                with open(train_config_path, 'w') as train_config_file:
                    json.dump(self.training_config, train_config_file, indent=2)

            # Save optimizer (if specified in fit or fit_wandb)
            if optimizer:
                opt_path = os.path.join(run_dir, f'{epoch} epochs optimizer.pth')
                torch.save(optimizer.state_dict(), opt_path)

            # Save scheduler (if specified in fit or fit_wandb)
            if scheduler:
                sched_path = os.path.join(run_dir, f'{epoch} epochs scheduler.pth')
                torch.save(scheduler.state_dict(), sched_path)

            # Save state dict
            model_path = os.path.join(run_dir, f'{epoch} epochs state dict.pth')
            torch.save(self.state_dict(), model_path)

    def compute_nmse(self, predicted, ground_truth):
        assert predicted.shape == ground_truth.shape, "Predicted and ground truth must have the same shape."

        numerator = torch.sum((predicted - ground_truth) ** 2, dim=[1, 2, 3])
        denominator = torch.sum(ground_truth ** 2, dim=[1, 2, 3])
        denominator = torch.clamp(denominator, min=1e-6)

        nmse = numerator / denominator
        return nmse.mean()            