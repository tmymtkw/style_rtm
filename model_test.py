print("code started")
from torchinfo import summary
from mmpose.models.backbones.custom_vit import EdgeViT
from mmpose.models.heads.heatmap_heads.custom_heads import CustomHead
from torch import nn

def main():
        backbone = EdgeViT(
                arch={
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 384 * 4
                },
                img_size=(256, 192),
                patch_size=16,
                qkv_bias=True,
                drop_path_rate=0.1,
                with_cls_token=True,
                out_type='featmap',
                patch_cfg=dict(padding=2)
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='checkpoints/mae_pretrain_vit_small.pth')
        )
        head = CustomHead(
                in_channels=384,
                out_channels=17,
                deconv_out_channels=(256, 256),
                deconv_kernel_sizes=(4, 4),
                loss=dict(type='KeypointMSELoss', use_target_weight=True),
                )
        model = nn.Sequential(
                backbone,
                head
        )
        summary(model,
                input_size=(1, 3, 256, 192),
                col_names=["output_size", "num_params"],
                )

        del model
        del backbone
        del head

if __name__ == '__main__':
        main()
