import torchvision
from lightning.pytorch import LightningModule
from nets.enc_dec import *
from nets.quantizer import *
from nets.cbk2cbk import *
from nets.blocks import PixelSNAIL
from utils.funcs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class KMEANVQ(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_channels = 3
        self.channel = args.n_hid * 2
        self.n_res_channel = args.n_hid
        self.n_res_block = 2
        self.embedding_dim = args.embedding_dim
        self.n_embed = args.num_embeddings

        self.encoder = Encoder(self.input_channels, self.channel, self.n_res_block, self.n_res_channel, stride=4)
        self.quantize_conv = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.quantize = EMAQuantizer(self.embedding_dim, self.n_embed)
        self.decoder = Decoder(self.embedding_dim, self.input_channels, self.channel, self.n_res_block, self.n_res_channel) #stride=4

        # DKM options
        self.cluster_target = args.cluster_target
        self.quantize_cluster = Quantizer(self.embedding_dim)

        self.criterion = nn.MSELoss()
        self.lr = args.lr

    def encode(self, input):
        enc = self.encoder(input)
        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
        quant, diff, ind = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)
        diff = diff.unsqueeze(0)
        return quant, diff, ind

    def decode(self, quant):
        out = self.decoder(quant)
        return out

    def forward(self, x):
        quant, diff, ind = self.encode(x)
        x_hat = self.decode(quant)
        return x_hat, diff, ind

    def clustering(self, x):
        enc = self.encoder(x)
        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
        
        # Kmeans codebook clustering
        if self.cluster_target < self.n_embed:
            # KMVQ
            clustered_weight, clustered_weight_idx = dkm(weights=self.quantize.embed.weight, k=self.cluster_target, args=self.args)
            quant, diff, ind = self.quantize_cluster(z_e=quant, embed_weight=clustered_weight)
            
            # # random select
            # clustered_weight = self.quantize.embed.weight[0:self.cluster_target] # random codebook reset
            # quant, diff, ind = self.quantize_cluster(z_e=quant, embed_weight=clustered_weight)

        # Inverse kmeans codebook clustering
        elif self.cluster_target > self.n_embed:
            quant, diff, ind = self.quantize_cluster(z_e=quant, embed_weight=self.args.i_clustered_weight.to(self.device))
        else:
            quant, diff, ind = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)
        diff = diff.unsqueeze(0)
        x_hat = self.decode(quant)
        return x_hat, diff, ind

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        x_hat, latent_loss, _ = self.forward(x)
        recon_loss = self.criterion(x_hat, x)
        loss = recon_loss + latent_loss
        self.log('train_recon_loss', recon_loss)
        self.log('train_latent_loss', latent_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        x_hat, latent_loss, ind_og = self.forward(x)
        recon_loss = self.criterion(x_hat, x)
        loss = recon_loss + latent_loss
        self.log('val_recon_loss', recon_loss)
        self.log('val_latent_loss', latent_loss)
        self.log('val_total_loss', loss)

        # # log sampled images
        # tensorboard = self.logger.experiment
        # grid = torchvision.utils.make_grid(x)
        # tensorboard.add_image('val_source_imnages', grid, self.global_step)
        # grid = torchvision.utils.make_grid(x_hat)
        # tensorboard.add_image('val_generated_images', grid, self.global_step)

        # encodings = F.one_hot(ind_og, self.quantize.n_embed).float().reshape(-1, self.quantize.n_embed)
        # avg_probs = encodings.mean(0)
        # perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        # cluster_use = torch.sum(avg_probs > 0, dtype=torch.float32)
        # self.log('val_perplexity_og', perplexity)
        # self.log('val_cluster_use_og', cluster_use)
        return loss

    def test_step(self, test_batch, batch_idx):
        x = test_batch[0]
        x_hat, _, ind = self.clustering(x)

        recon_loss = self.criterion(x_hat, x)
        self.log('test_recon_loss', recon_loss, prog_bar=True)

        # # log sampled images
        # tensorboard = self.logger.experiment
        # grid = torchvision.utils.make_grid(x)
        # tensorboard.add_image('test_source_imnages', grid, self.global_step)
        # grid = torchvision.utils.make_grid(x_hat)
        # tensorboard.add_image('test_generated_images', grid, self.global_step)

        psnr = 10 * torch.log10((1**2)/recon_loss)
        self.log('test_PSNR', psnr, prog_bar=True)

        ms_ssim_val = ssim(x, x_hat, data_range=1, size_average=True)
        self.log('test_SSIM', ms_ssim_val, prog_bar=True)

        encodings = F.one_hot(ind, self.cluster_target).float().reshape(-1, self.cluster_target)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity', perplexity, prog_bar=True)
        self.log('test_cluster_use', cluster_use, prog_bar=True)

    def decode_latent(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class VRVQ(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_channels = 3
        self.channel = args.n_hid * 2
        self.n_res_channel = args.n_hid
        self.n_res_block = 2
        self.embedding_dim = args.embedding_dim
        self.n_embed = args.num_embeddings
        self.n_embed_min = args.num_embeddings_min
        self.n_embed_max = args.num_embeddings_max
        self.n_embed_test = args.num_embeddings_test

        self.encoder = Encoder(self.input_channels, self.channel, self.n_res_block, self.n_res_channel, stride=4)
        self.quantize_conv = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.quantize = EMAQuantizer(self.embedding_dim, self.n_embed)
        self.cbk_enc = CdBkEncoder(self.embedding_dim, self.embedding_dim, 2, 0.5)
        self.cbk_dec = CdBkDecoder(self.embedding_dim, self.embedding_dim, 2, 0.5)
        self.embed_layer_dec = nn.Embedding(self.n_embed_max, self.embedding_dim)
        self.cbk2cbk = CdBk2CdBk(self.cbk_enc, self.cbk_dec, self.quantize.embed, self.embed_layer_dec, self.args.device)
        self.decoder = Decoder(self.embedding_dim, self.input_channels, self.channel, self.n_res_block, self.n_res_channel) #stride=4

        self.quantize_cbk = Quantizer(self.embedding_dim)
        self.src = torch.arange(self.n_embed).unsqueeze(1)

        self.criterion = nn.MSELoss()
        self.lr = args.lr

    def encode(self, input, trg):
        enc = self.encoder(input)
        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)

        # source
        quant_src, diff_src, ind_src = self.quantize(quant)
        quant_src = quant_src.permute(0, 3, 1, 2)
        diff_src = diff_src.unsqueeze(0)

        # cbk2cbk
        embed_weight = self.cbk2cbk(self.src.to(self.device), trg.to(self.device)).squeeze(1)
        quant_trg, diff_trg, ind_trg = self.quantize_cbk(quant, embed_weight.to(self.device))
        quant_trg = quant_trg.permute(0, 3, 1, 2)
        diff_trg = diff_trg.unsqueeze(0)
        return quant_src, diff_src, ind_src, quant_trg, diff_trg, ind_trg

    def decode(self, quant):
        out = self.decoder(quant)
        return out

    def forward(self, x, trg):
        quant_src, diff_src, ind_src, quant_trg, diff_trg, ind_trg = self.encode(x, trg)
        x_hat_src = self.decode(quant_src)
        x_hat_trg = self.decode(quant_trg)
        return x_hat_src, diff_src, ind_src, x_hat_trg, diff_trg, ind_trg

    def sample_train_trg(self, num_min, num_max):
        num_trg = random.randint(num_min, num_max)
        trg = torch.arange(num_trg).unsqueeze(1)
        return trg

    def sample_val_trg(self, num_min, num_max):
        min_exponent = int(math.log2(num_min))
        max_exponent = int(math.log2(num_max))
        random_exponent = random.randint(min_exponent, max_exponent)
        num_trg = 2 ** random_exponent
        trg = torch.arange(num_trg).unsqueeze(1)
        return trg

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        trg = self.sample_train_trg(self.n_embed_min, self.n_embed_max)

        x_hat_src, latent_loss_src, _, x_hat_trg, latent_loss_trg, _ = self.forward(x, trg)

        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)
        latent_loss = latent_loss_src + latent_loss_trg
        loss = recon_loss + latent_loss
        self.log('train_recon_loss', recon_loss)
        self.log('train_latent_loss', latent_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_ster(self, val_batch, batch_idx):
        x = val_batch[0]
        trg = self.sample_val_trg(self.n_embed_min, self.n_embed_max)

        x_hat_src, latent_loss_src, _, x_hat_trg, latent_loss_trg, _ = self.forward(x, trg)

        recon_loss_src = self.criterion(x_hat_src, x)
        recon_loss_trg = self.criterion(x_hat_trg, x)
        self.log('val_recon_loss', recon_loss_src + recon_loss_trg)
        self.log('val_recon_loss(src)', recon_loss_src)
        self.log('val_recon_loss(trg)', recon_loss_trg)

        # log sampled images
        tensorboard = self.logger.experiment
        grid = torchvision.utils.make_grid(x)
        tensorboard.add_image('val_source_imnages', grid, self.global_step)
        grid = torchvision.utils.make_grid(x_hat_trg)
        tensorboard.add_image('val_generated_images', grid, self.global_step)

    def test_step(self, test_batch, batch_idx):
        x = test_batch[0]
        trg_test = torch.arange(self.n_embed_test).unsqueeze(1)
        x_hat_src, latent_loss_src, ind_src, x_hat_trg, latent_loss_trg, ind_trg = self.forward(x, trg_test)
        if self.n_embed == self.n_embed_test:
            x_hat_trg = x_hat_src
            ind_trg = ind_src

        recon_loss = self.criterion(x_hat_trg, x)
        self.log('test_recon_loss', recon_loss, prog_bar=True)

        psnr = 10 * torch.log10((1**2)/recon_loss)
        self.log('test_PSNR', psnr, prog_bar=True)

        ms_ssim_val = ssim(x, x_hat_trg, data_range=1, size_average=True)
        self.log('test_SSIM', ms_ssim_val, prog_bar=True)

        # # log sampled images
        # tensorboard = self.logger.experiment
        # grid = torchvision.utils.make_grid(x)
        # tensorboard.add_image('test_source_imnages', grid, self.global_step)
        # grid = torchvision.utils.make_grid(x_hat_trg)
        # tensorboard.add_image('test_generated_images', grid, self.global_step)

        encodings = F.one_hot(ind_trg, self.n_embed_test).float().reshape(-1, self.n_embed_test)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity', perplexity, prog_bar=True)
        self.log('test_cluster_use', cluster_use, prog_bar=True)

    def decode_latent(self, code, trg):
        quant = F.embedding(code, self.cbk2cbk(self.src.to(self.device), trg.to(self.device)).squeeze(1).to(self.device))
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class PixelSNAIL_ONE(LightningModule):
    def __init__(self, n_codes, n_filters, n_res_blocks, n_snail_blocks, n_condition_blocks, args):
        super().__init__()
        self.args = args
        self.bottom = PixelSNAIL(attention=False, input_channels=n_codes, n_codes=n_codes,
                                 n_filters=n_filters, n_res_blocks=n_res_blocks,
                                 n_snail_blocks=n_snail_blocks)
        self.vqvae = VRVQ.load_from_checkpoint("../saved_model/vrvq/" + str(args.dataset)
                                           +  "/base_voca" + str(args.num_embeddings)
                                           + "/seed" + str(args.seed) + '/model.ckpt'
                                           , args=args)
        self.condition_stack = []

        for _ in range(n_condition_blocks):
            self.condition_stack.extend([
                nn.Conv2d(in_channels=n_codes, out_channels=n_codes, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv2d(in_channels=n_codes, out_channels=n_codes, kernel_size=3, padding=1)
            ])

        self.condition_stack = nn.ModuleList(
            [nn.Conv2d(in_channels=n_codes, out_channels=n_codes, kernel_size=3, padding=1)
             for _ in range(n_condition_blocks)])
    
        self.criterion = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()
        self.n_embed_test = args.num_embeddings_test
    

    def forward(self, x):
        out = self.bottom(x)
        return out

    def to_one_hot(self, input):
        return F.one_hot(input, num_classes=self.n_embed_test).permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.vqvae.eval()
        x = train_batch[0]
        trg_test = torch.arange(self.n_embed_test).unsqueeze(1)
        _, _, _, _, _, ind = self.vqvae.encode(x, trg_test)
        
        code = self.to_one_hot(ind)
        out_pixel = self.forward(code)
        loss = self.criterion(out_pixel, ind)
        self.log('train_loss', loss, prog_bar=True)

        # ind_pixel = torch.argmax(out_pixel, dim=1)
        # # in_bottom = train_batch
        # # x = in_bottom.type(torch.FloatTensor).to(self.device)
        # # y = torch.argmax(x, dim=1)
        # # x_hat = self.forward(x)
        # # loss = self.criterion(x_hat, y)
        # # self.log('pixelcnn_train_loss', loss)
        # # y_hat = torch.argmax(x_hat, dim=1)
        # decoded_sample_before = self.vqvae.decode_latent(ind, trg_test)
        # decoded_sample_after = self.vqvae.decode_latent(ind_pixel, trg_test)
        # 
        # recon_loss = self.criterion(decoded_sample_after, decoded_sample_before)
        # self.log('train_recon_loss', recon_loss, prog_bar=True)
        # 
        # psnr = 10 * torch.log10((1**2)/recon_loss)
        # self.log('train_PSNR', psnr, prog_bar=True)
        # 
        # ms_ssim_val = ssim(decoded_sample_before, decoded_sample_after, data_range=1, size_average=True)
        # self.log('train_SSIM', ms_ssim_val, prog_bar=True)
        # 
        # # log sampled images
        # tensorboard = self.logger.experiment
        # grid = torchvision.utils.make_grid(decoded_sample_before)
        # tensorboard.add_image('train_before pixel', grid, self.global_step)
        # grid = torchvision.utils.make_grid(decoded_sample_after)
        # tensorboard.add_image('train_after pixel', grid, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        trg_test = torch.arange(self.n_embed_test).unsqueeze(1)
        _, _, _, _, _, ind = self.vqvae.encode(x, trg_test)
        code = self.to_one_hot(ind)
        out_pixel = self.forward(code)

        loss = self.criterion(out_pixel, ind)
        self.log('val_loss', loss, prog_bar=True)
        ind_pixel = torch.argmax(out_pixel, dim=1)
        # decoded_sample_before = self.vqvae.decode_latent(ind, trg_test)
        decoded_sample_after = self.vqvae.decode_latent(ind_pixel, trg_test)
        
        recon_loss = self.criterion(decoded_sample_after, x)
        self.log('val_recon_loss', recon_loss, prog_bar=True)

        psnr = 10 * torch.log10((1**2)/recon_loss)
        self.log('val_PSNR', psnr, prog_bar=True)

        ms_ssim_val = ssim(x, decoded_sample_after, data_range=1, size_average=True)
        self.log('val_SSIM', ms_ssim_val, prog_bar=True)
        
        # # log sampled images
        # tensorboard = self.logger.experiment
        # grid = torchvision.utils.make_grid(x)
        # tensorboard.add_image('val_before pixel', grid, self.global_step)
        # grid = torchvision.utils.make_grid(decoded_sample_after)
        # tensorboard.add_image('val_after pixel', grid, self.global_step)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x = test_batch[0]
        trg_test = torch.arange(self.n_embed_test).unsqueeze(1)
        _, _, _, _, _, ind = self.vqvae.encdoe(x, trg_test)
        code = self.to_one_hot(ind)
        out_pixel = self.forward(code)

        loss = self.criterion(out_pixel, ind)
        self.log('test_loss', loss, prog_bar=True)
        ind_pixel = torch.argmax(out_pixel, dim=1)
        decoded_sample_before = self.vqvae.decode_latent(ind, trg_test)
        decoded_sample_after = self.vqvae.decode_latent(ind_pixel, trg_test)

        recon_loss = self.criterion(decoded_sample_after, x)
        self.log('test_recon_loss', recon_loss, prog_bar=True)

        psnr = 10 * torch.log10((1**2)/recon_loss)
        self.log('test_PSNR', psnr, prog_bar=True)

        ms_ssim_val = ssim(x, decoded_sample_after, data_range=1, size_average=True)
        self.log('test_SSIM', ms_ssim_val, prog_bar=True)

        # log sampled images
        tensorboard = self.logger.experiment
        grid = torchvision.utils.make_grid(decoded_sample_before)
        tensorboard.add_image('test_before pixel', grid, self.global_step)
        grid = torchvision.utils.make_grid(decoded_sample_after)
        tensorboard.add_image('test_after pixel', grid, self.global_step)
        return loss