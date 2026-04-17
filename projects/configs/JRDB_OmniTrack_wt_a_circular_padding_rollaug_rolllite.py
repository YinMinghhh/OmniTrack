_base_ = ["./JRDB_OmniTrack_wt_a_circular_padding_rollaug.py"]

data = dict(
    train=dict(
        data_aug_conf=dict(
            roll_prob=0.25,
        )
    )
)
