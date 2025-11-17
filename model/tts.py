import torch

from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, fix_len_compatibility


class GradTTSWithEmo(BaseModule):
    """Inference-focused Grad-TTS with speaker and emotion conditioning."""

    def __init__(
        self,
        n_vocab=148,
        n_spks=1,
        n_emos=5,
        spk_emb_dim=64,
        n_enc_channels=192,
        filter_channels=768,
        filter_channels_dp=256,
        n_heads=2,
        n_enc_layers=6,
        enc_kernel=3,
        enc_dropout=0.1,
        window_size=4,
        n_feats=80,
        dec_dim=64,
        beta_min=0.05,
        beta_max=20.0,
        pe_scale=1000,
        use_classifier_free=False,
        dummy_spk_rate=0.5,
        **kwargs,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.n_emos = n_emos
        self.spk_emb_dim = spk_emb_dim
        self.use_classifier_free = use_classifier_free

        self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.emo_emb = torch.nn.Embedding(n_emos, spk_emb_dim)
        self.merge_spk_emo = torch.nn.Sequential(
            torch.nn.Linear(spk_emb_dim * 2, spk_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(spk_emb_dim, spk_emb_dim),
        )

        self.encoder = TextEncoder(
            n_vocab,
            n_feats,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
            spk_emb_dim=spk_emb_dim,
            n_spks=n_spks,
        )
        self.decoder = Diffusion(
            n_feats,
            dec_dim,
            spk_emb_dim,
            beta_min,
            beta_max,
            pe_scale,
        )

        if self.use_classifier_free:
            self.dummy_xv = torch.nn.Parameter(torch.randn(size=(spk_emb_dim,)))
            self.dummy_rate = dummy_spk_rate
            print(f"Using classifier free with rate {self.dummy_rate}")

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        emo=None,
        length_scale=1.0,
        classifier_free_guidance=1.0,
        force_dur=None,
    ):
        """Generate mel-spectrograms for the provided text batch."""

        x, x_lengths = self.relocate_input([x, x_lengths])

        spk = self.spk_emb(spk)
        emo = self.emo_emb(emo)

        if self.use_classifier_free:
            emo = emo / torch.sqrt(torch.sum(emo**2, dim=1, keepdim=True))

        spk_merged = self.merge_spk_emo(torch.cat([spk, emo], dim=-1))

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        if force_dur is not None:
            w_ceil = force_dur.unsqueeze(1)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        
        attn = generate_path(w_ceil.squeeze(1), attn_mask)

        mu_y = torch.matmul(mu_x, attn.transpose(1, 2))
        encoder_outputs = mu_y[:, :, :y_max_length]

        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature

        unit_dummy_emo = None
        if self.use_classifier_free:
            unit_dummy_emo = self.dummy_xv / torch.sqrt(torch.sum(self.dummy_xv**2))

        dummy_spk = None
        if self.use_classifier_free:
            repeated_dummy = unit_dummy_emo.unsqueeze(0).repeat(len(spk), 1)
            dummy_spk = self.merge_spk_emo(torch.cat([spk, repeated_dummy], dim=-1))

        decoder_outputs = self.decoder(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            stoc,
            spk_merged,
            use_classifier_free=self.use_classifier_free,
            classifier_free_guidance=classifier_free_guidance,
            dummy_spk=dummy_spk,
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn.unsqueeze(1)[:, :, :y_max_length]
