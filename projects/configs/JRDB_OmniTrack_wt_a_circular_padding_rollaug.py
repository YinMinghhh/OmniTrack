_base_ = ["./JRDB_OmniTrack.py"]

model = dict(
    img_neck=dict(
        circular_padding_cfg=dict(
            enabled=True,
            conv3x3=True,
            repc3=True,
        )
    )
)

data = dict(
    train=dict(
        data_aug_conf=dict(
            roll_prob=1.0,
            roll_px_range=(0, 3759),
            roll_stride=1,
        )
    )
)
