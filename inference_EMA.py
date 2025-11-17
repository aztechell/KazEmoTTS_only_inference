import json
import re
from datetime import datetime
from pathlib import Path

import torch
from attrdict import AttrDict
from num2words import num2words
from pydub import AudioSegment

import utils_data as utils
from model import GradTTSWithEmo
from models import Generator as HiFiGAN
from text import convert_text

HIFIGAN_CONFIG = './configs/hifigan-config.json'
HIFIGAN_CHECKPT = r'.\\pre_trained\\g_01720000'


if __name__ == '__main__':
    hps, args = utils.get_hparams_decode()
    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    ckpt = args.model
    model = GradTTSWithEmo(**hps.model).to(device)
    utils.load_checkpoint(ckpt, model, None)
    _ = model.cuda().eval()

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    emos = sorted(["angry", "surprise", "fear", "happy", "neutral", "sad"])
    speakers = ['M1', 'F1', 'M2']

    entries = []
    with open(args.file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                print(f"Skip line with unexpected format: {line}")
                continue
            text, emo_idx, spk_idx = parts
            try:
                emo_id = int(emo_idx)
                spk_id = int(spk_idx)
            except ValueError:
                print(f"Skip line with non-integer emotion/speaker ids: {line}")
                continue
            if not 0 <= emo_id < len(emos):
                print(f"Skip line with out-of-range emotion id: {line}")
                continue
            if not 0 <= spk_id < len(speakers):
                print(f"Skip line with out-of-range speaker id: {line}")
                continue
            text = re.sub(r'(\d+)', lambda m: num2words(m.group(), lang='kz'), text)
            entries.append((text, emo_id, spk_id))

    for text, emo_i, control_spk_id in entries:
        control_emo_id = emo_i
        with torch.no_grad():
            ### define emotion
            emo = torch.LongTensor([control_emo_id]).to(device)
            sid = torch.LongTensor([control_spk_id]).to(device)
            text_padded, text_len = convert_text(text)
            y_enc, y_dec, attn = model.forward(
                text_padded,
                text_len,
                n_timesteps=args.timesteps,
                temperature=args.noise,
                stoc=args.stoc,
                spk=sid,
                emo=emo,
                length_scale=1.,
                classifier_free_guidance=args.guidance,
            )
        res = y_dec.squeeze().cpu().numpy()
        x = torch.from_numpy(res).unsqueeze(0).cuda()
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio = audio.detach().cpu().numpy().astype('int16')
        audio = AudioSegment(audio.data, frame_rate=22050, sample_width=2, channels=1)

        out_dir = Path(args.generated_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        spk = speakers[control_spk_id]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{emos[emo_i]}_{spk}_{ts}.mp3"
        audio.export(str(out_dir / fname), format="mp3", bitrate="192k")

        del y_enc, y_dec, attn, audio
        torch.cuda.empty_cache()
