_base_ = ["./JRDB_OmniTrack.py"]

model = dict(
    head=dict(
        instance_bank=dict(
            tracking_mode="tbd",
            tbd_backend="hybridsort",
            tbd_handler_cfg=dict(
                pre_tracker_quantize_boxes=False,
            ),
            tbd_tracker_cfg=dict(
                use_circular_track=True,
            ),
        ),
    )
)
