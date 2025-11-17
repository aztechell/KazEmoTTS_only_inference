import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    Generates an alignment path from durations.
    duration: [b, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device
    b, _, t_y, t_x = mask.shape
    
    path = torch.zeros(b, t_y, t_x, dtype=mask.dtype).to(device)
    duration_int = torch.round(duration).long()
    
    for i in range(b):
        cum_duration = torch.cumsum(duration_int[i], -1)
        
        for j in range(t_x):
            start_time = cum_duration[j-1] if j > 0 else 0
            end_time = cum_duration[j]
            
            if start_time < end_time:
                start = min(start_time, t_y)
                end = min(end_time, t_y)
                # The original code had the dimensions swapped.
                # We need to align the text tokens (t_x) with the mel frames (t_y).
                path[i, start:end, j] = 1
                
    return path
