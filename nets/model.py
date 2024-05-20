import torchvision
import pytorch_lightning as pl
from nets.enc_dec import *
from nets.quantizer import *
from nets.cbk2cbk import *
from nets.blocks import PixelSNAIL
from utils.funcs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class VQVAE_ONE(pl.LightningModule):
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

    def forward_clustering(self, x):
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
            clustered_weight = self.args.i_clustered_weight.to(self.device)
            quant, diff, ind = self.quantize_cluster(z_e=quant, embed_weight=clustered_weight)
        else:
            clustered_weight = self.quantize.embed.weight
            quant, diff, ind = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)
        diff = diff.unsqueeze(0)
        x_hat = self.decode(quant)
        return x_hat, diff, ind, clustered_weight

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
        x_hat, _, ind, _ = self.forward_clustering(x)

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

    def decode_latent(self, code, weight):
        quant = F.embedding(code, weight)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class RAQVAE_ONE(pl.LightningModule):
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

    def validation_step(self, val_batch, batch_idx):
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

        # log sampled images
        tensorboard = self.logger.experiment
        grid = torchvision.utils.make_grid(x)
        tensorboard.add_image('test_source_imnages', grid, self.global_step)
        grid = torchvision.utils.make_grid(x_hat_trg)
        tensorboard.add_image('test_generated_images', grid, self.global_step)

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


class VQVAE_TWO(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_channels = 3
        self.channel = args.n_hid * 2
        self.n_res_channel = args.n_hid
        self.n_res_block = 2
        self.embedding_dim = args.embedding_dim
        self.n_embed = args.num_embeddings
        
        self.encoder_b = Encoder(self.input_channels, self.channel, self.n_res_block, self.n_res_channel, stride=4)
        self.encoder_t = Encoder(self.channel, self.channel, self.n_res_block, self.n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.quantize_conv_t = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.upsample_t = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, 4, stride=2, padding=1)
    
        self.quantize = EMAQuantizer(self.embedding_dim, self.n_embed)
        self.decoder = Decoder(self.embedding_dim+self.embedding_dim, self.input_channels, self.channel, self.n_res_block, self.n_res_channel) #stride=4

        # DKM options
        self.cluster_target = args.cluster_target
        self.quantize_cluster = Quantizer(self.embedding_dim)

        self.criterion = nn.MSELoss()
        self.lr = args.lr

    def encode(self, input):
        enc_b = self.encoder_b(input)
        enc_t = self.encoder_t(enc_b)
        
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, ind_b = self.quantize(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, ind_t = self.quantize(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        
        return quant_t, quant_b, diff_t+diff_b, ind_t, ind_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        out = self.decoder(quant)
        return out

    def forward(self, x):
        quant_t, quant_b, diff, ind_t, ind_b = self.encode(x)
        x_hat = self.decode(quant_t, quant_b)
        return x_hat, diff, ind_t, ind_b

    def forward_clustering(self, x):
        enc_b = self.encoder_b(x)
        enc_t = self.encoder_t(enc_b)
        
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
    
        # Kmeans codebook clustering
        if self.cluster_target < self.n_embed:
            # KMVQ
            clustered_weight, clustered_weight_idx = dkm(weights=self.quantize.embed.weight, k=self.cluster_target, args=self.args)
            quant_b, diff_b, ind_b = self.quantize_cluster(z_e=quant_b, embed_weight=clustered_weight)
            quant_t, diff_t, ind_t = self.quantize_cluster(z_e=quant_t, embed_weight=clustered_weight)
            
            # # random select
            # clustered_weight = self.quantize.embed.weight[0:self.cluster_target] # random codebook reset
            # quant, diff, ind = self.quantize_cluster(z_e=quant, embed_weight=clustered_weight)

        # Inverse kmeans codebook clustering
        elif self.cluster_target > self.n_embed:
            clustered_weight = self.args.i_clustered_weight.to(self.device)
            quant_b, diff_b, ind_b = self.quantize_cluster(z_e=quant_b, embed_weight=clustered_weight)
            quant_t, diff_t, ind_t = self.quantize_cluster(z_e=quant_t, embed_weight=clustered_weight)
        else:
            clustered_weight = self.quantize.embed.weight
            quant_b, diff_b, ind_b = self.quantize(quant_b)
            quant_t, diff_t, ind_t = self.quantize(quant_t)
            
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        x_hat = self.decode(quant_t, quant_b)
        return x_hat, diff_b+diff_t, ind_t, ind_b, clustered_weight

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        x_hat, latent_loss, _, _ = self.forward(x)
        recon_loss = self.criterion(x_hat, x)
        loss = recon_loss + latent_loss
        self.log('train_recon_loss', recon_loss)
        self.log('train_latent_loss', latent_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        x_hat, latent_loss,  _, _ = self.forward(x)
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
        x_hat, _, ind_t, ind_b, _ = self.forward_clustering(x)

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

        encodings = F.one_hot(ind_t, self.cluster_target).float().reshape(-1, self.cluster_target)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity_top', perplexity, prog_bar=True)
        self.log('test_cluster_use_top', cluster_use, prog_bar=True)
        encodings = F.one_hot(ind_b, self.cluster_target).float().reshape(-1, self.cluster_target)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity_bottom', perplexity, prog_bar=True)
        self.log('test_cluster_use_bottom', cluster_use, prog_bar=True)

    def decode_latent(self, code, weight):
        quant = F.embedding(code, weight)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    
class RAQVAE_TWO(pl.LightningModule):
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

        self.encoder_b = Encoder(self.input_channels, self.channel, self.n_res_block, self.n_res_channel, stride=4)
        self.encoder_t = Encoder(self.channel, self.channel, self.n_res_block, self.n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.quantize_conv_t = nn.Conv2d(self.channel, self.embedding_dim, 1)
        self.quantize = EMAQuantizer(self.embedding_dim, self.n_embed)
        self.upsample_t = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, 4, stride=2, padding=1)
        self.decoder = Decoder(self.embedding_dim+self.embedding_dim, self.input_channels, self.channel, self.n_res_block, self.n_res_channel) #stride=4
    
        # Seq2Seq
        self.cbk_enc = CdBkEncoder(self.embedding_dim, self.embedding_dim, 2, 0.5)
        self.cbk_dec = CdBkDecoder(self.embedding_dim, self.embedding_dim, 2, 0.5)
        self.embed_layer_dec = nn.Embedding(self.n_embed_max, self.embedding_dim)
        self.cbk2cbk = CdBk2CdBk(self.cbk_enc, self.cbk_dec, self.quantize.embed, self.embed_layer_dec, self.args.device)
        self.quantize_cbk = Quantizer(self.embedding_dim)
        self.src = torch.arange(self.n_embed).unsqueeze(1)

        self.criterion = nn.MSELoss()
        self.lr = args.lr
    
    def forward(self, x, trg):
        quant_src_t, quant_src_b, diff_src, ind_src_t, ind_src_b, quant_trg_t, quant_trg_b, diff_trg, ind_trg_t, ind_trg_b = self.encode(x, trg)
        
        x_hat_src = self.decode(quant_src_t, quant_src_b)
        x_hat_trg = self.decode(quant_trg_t, quant_trg_b)
        
        return x_hat_src, diff_src, ind_src_t, ind_src_b, x_hat_trg, diff_trg, ind_trg_t, ind_trg_b

    def encode(self, input, trg):
        enc_b = self.encoder_b(input)
        enc_t = self.encoder_t(enc_b)
        
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_src_b, diff_src_b, ind_src_b = self.quantize(quant_b)
        quant_src_b = quant_src_b.permute(0, 3, 1, 2)
        diff_src_b = diff_src_b.unsqueeze(0)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_src_t, diff_src_t, ind_src_t = self.quantize(quant_t)
        quant_src_t = quant_src_t.permute(0, 3, 1, 2)
        diff_src_t = diff_src_t.unsqueeze(0)
        
        # cbk2cbk
        embed_weight = self.cbk2cbk(self.src.to(self.device), trg.to(self.device)).squeeze(1)
        quant_trg_b, diff_trg_b, ind_trg_b = self.quantize_cbk(quant_b, embed_weight.to(self.device))
        quant_trg_b = quant_trg_b.permute(0, 3, 1, 2)
        diff_trg_b = diff_trg_b.unsqueeze(0)
        quant_trg_t, diff_trg_t, ind_trg_t = self.quantize_cbk(quant_t, embed_weight.to(self.device))
        quant_trg_t = quant_trg_t.permute(0, 3, 1, 2)
        diff_trg_t = diff_trg_t.unsqueeze(0)
        
        return quant_src_t, quant_src_b, diff_src_b+diff_src_t, ind_src_t, ind_src_b, quant_trg_t, quant_trg_b, diff_trg_b+diff_trg_t, ind_trg_t, ind_trg_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        out = self.decoder(quant)
        return out

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

        x_hat_src, latent_loss_src, _, _, x_hat_trg, latent_loss_trg, _, _ = self.forward(x, trg)

        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)
        latent_loss = latent_loss_src + latent_loss_trg
        loss = recon_loss + latent_loss
        self.log('train_recon_loss', recon_loss)
        self.log('train_latent_loss', latent_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        trg = self.sample_val_trg(self.n_embed_min, self.n_embed_max)

        x_hat_src, latent_loss_src, _, _, x_hat_trg, latent_loss_trg, _, _ = self.forward(x, trg)

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
        x_hat_src, latent_loss_src, ind_src_t, ind_src_b, x_hat_trg, latent_loss_trg, ind_trg_t, ind_trg_b = self.forward(x, trg_test)
        
        if self.n_embed == self.n_embed_test:
            x_hat_trg = x_hat_src
            ind_trg_t = ind_src_t
            ind_trg_b = ind_src_b

        recon_loss = self.criterion(x_hat_trg, x)
        self.log('test_recon_loss', recon_loss, prog_bar=True)

        psnr = 10 * torch.log10((1**2)/recon_loss)
        self.log('test_PSNR', psnr, prog_bar=True)

        ms_ssim_val = ssim(x, x_hat_trg, data_range=1, size_average=True)
        self.log('test_SSIM', ms_ssim_val, prog_bar=True)

        # log sampled images
        tensorboard = self.logger.experiment
        grid = torchvision.utils.make_grid(x)
        tensorboard.add_image('test_source_imnages', grid, self.global_step)
        grid = torchvision.utils.make_grid(x_hat_trg)
        tensorboard.add_image('test_generated_images', grid, self.global_step)

        encodings = F.one_hot(ind_trg_t, self.n_embed_test).float().reshape(-1, self.n_embed_test)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity_top', perplexity, prog_bar=True)
        self.log('test_cluster_use_top', cluster_use, prog_bar=True)
        encodings = F.one_hot(ind_trg_b, self.n_embed_test).float().reshape(-1, self.n_embed_test)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-9)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0, dtype=torch.float64)
        self.log('test_perplexity_bottom', perplexity, prog_bar=True)
        self.log('test_cluster_use_bottom', cluster_use, prog_bar=True)

    def decode_latent(self, code_t, code_b, trg):
        quant_t = F.embedding(code_t, self.cbk2cbk(self.src.to(self.device), trg.to(self.device)).squeeze(1).to(self.device))
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = F.embedding(code_b, self.cbk2cbk(self.src.to(self.device), trg.to(self.device)).squeeze(1).to(self.device))
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_t, quant_b)

        return dec

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
