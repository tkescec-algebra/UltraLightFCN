ENCODER_PARAMS = {
    # Encoder
    'enc_channels': [16, 16, 32, 32, 64],
    'enc_kernel_sizes': [3, 3, 3, 3, 3],
    'enc_strides': [1, 2, 2, 1, 1],
    'dilations': [2, 4],
    # Context
    'mini_aspp': True,
    'mini_aspp_gpool': True,
    # Attention
    'use_sa': True,
    'sa_windowed': True,
    'sa_window_size': 16,
    'sa_shifted': True,
    'sa_heads': 4,
    'sa_dropout': 0.1,
}