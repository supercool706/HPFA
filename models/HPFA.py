
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from .basic_layers import Transformer, CrossTransformer, HhyperLearningEncoder, GradientReversalLayer
from .bert import BertTextEncoder


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Sun(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(Sun, self).__init__()
        self.pool_c = nn.AdaptiveAvgPool1d(1)
        self.pool_n = nn.AdaptiveAvgPool1d(1)


        self.conv1 = nn.Conv1d(136, 136, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(136)
        self.act = HSwish()

        self.conv_h = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv1d(1, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        # x shape: [Batch, Tokens, Channels]

        # Channel pooling
        x_h = self.pool_c(x)  # [B, N, 1]

        # Token pooling (permute first)
        d = x.permute(0, 2, 1)
        x_w = self.pool_n(d)  # [B, C, 1]

        # Concatenate features
        y = torch.cat([x_h, x_w], dim=1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h = y[:, :8, :]
        x_w = y[:, :128, :]
        x_w = x_w.permute(0, 2, 1)

        a_h = x_h.sigmoid()
        a_w = x_w.sigmoid()

        out = identity * a_w * a_h
        return out

class DualPathInteraction(nn.Module):

    def __init__(self, token_dim, channel_dim, hidden_dim, depth=2, dropout=0.5):
        super(DualPathInteraction, self).__init__()
        self.depth = depth

        # Shared weights used across all depth iterations
        self.token_mixer = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(channel_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channel_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x_token = x.permute(0, 2, 1)
            mixed_token = self.token_mixer(x_token)
            x = x + mixed_token.permute(0, 2, 1)
            x = x + self.channel_mixer(x)

        return x

class TemporalSegmentAttention(nn.Module):

    def __init__(self, in_channels, block_channel, block_dropout=0., lowest_atten=0., reduction_factor=4,
                 split_block=1):
        super(TemporalSegmentAttention, self).__init__()
        self.in_channels = in_channels
        self.block_channel = block_channel
        self.dropout_rate = block_dropout
        self.lowest_atten = lowest_atten
        self.split_block = split_block
        self.num_modality = len(in_channels)


        self.reduced_channel = self.block_channel // reduction_factor
        self.num_blocks = [math.ceil(ic / self.block_channel) for ic in in_channels]
        self.last_block_padding = [ic % self.block_channel for ic in in_channels]
        self.total_blocks = sum(self.num_blocks)

        self.softmax = nn.Softmax(dim=0)


        self.segment_layers = nn.ModuleList()
        for _ in range(split_block):
            segment_dict = nn.ModuleDict({
                'joint_projector': nn.Sequential(
                    nn.Linear(self.block_channel, self.reduced_channel),
                    nn.BatchNorm1d(self.reduced_channel),
                    nn.ReLU(inplace=True)
                ),
                'attention_heads': nn.ModuleList([
                    nn.Linear(self.reduced_channel, self.block_channel)
                    for _ in range(self.total_blocks)
                ])
            })
            self.segment_layers.append(segment_dict)

    def _generate_mask(self, x_list):

        blocks_per_mod = [x.shape[1] for x in x_list]
        total_blocks = sum(blocks_per_mod)
        batch_size = x_list[0].shape[0]

        mask_dist = torch.distributions.binomial.Binomial(probs=1 - self.dropout_rate)
        raw_mask = mask_dist.sample(torch.Size([batch_size, total_blocks])).to(x_list[0].device)

        scaled_mask = raw_mask * (1.0 / (1 - self.dropout_rate))

        mask_shapes = [list(x.shape[:2]) + [1] * (x.dim() - 2) for x in x_list]
        grouped_masks = torch.split(scaled_mask, blocks_per_mod, dim=1)
        grouped_masks = [m.reshape(s) for m, s in zip(grouped_masks, mask_shapes)]

        return grouped_masks

    def _process_segment(self, inputs, layer_idx):

        layers = self.segment_layers[layer_idx]
        joint_proj = layers['joint_projector']
        atten_heads = layers['attention_heads']

        padded_inputs = [F.pad(x, (0, pad)) for pad, x in zip(self.last_block_padding, inputs)]


        reshaped_inputs = []
        for x, nb in zip(padded_inputs, self.num_blocks):
            target_shape = [x.shape[0], nb, self.block_channel] + list(x.shape[2:])
            reshaped_inputs.append(x.reshape(target_shape))


        masks = None
        if self.training and 0 < self.dropout_rate < 1:
            masks = self._generate_mask(reshaped_inputs)
            reshaped_inputs_drop = [x * m for x, m in zip(reshaped_inputs, masks)]
        else:
            reshaped_inputs_drop = reshaped_inputs


        descriptors = []
        for x in reshaped_inputs_drop:
            block_sum = x.sum(dim=1)
            flat = block_sum.view(block_sum.size(0), block_sum.size(1), -1)
            gap = F.adaptive_avg_pool1d(flat, 1).squeeze(-1)
            descriptors.append(gap)

        global_descriptor = torch.stack(descriptors).sum(dim=0)
        global_descriptor = joint_proj(global_descriptor)

        scores = [head(global_descriptor) for head in atten_heads]
        atten_weights = self.softmax(torch.stack(scores)).permute(1, 0, 2)
        atten_weights = self.lowest_atten + atten_weights * (1 - self.lowest_atten)

        atten_splits = torch.split(atten_weights, self.num_blocks, dim=1)

        outputs = []
        for i, (x_feat, att) in enumerate(zip(reshaped_inputs, atten_splits)):
            att_shape = list(att.shape) + [1] * (x_feat.dim() - 3)
            att = att.reshape(att_shape)

            if self.training and masks is not None:
                out = x_feat * masks[i] * att
            else:
                out = x_feat * att

            outputs.append(out.reshape(inputs[i].shape))

        return outputs

    def forward(self, inputs):
        for x, ic in zip(inputs, self.in_channels):
            if x.shape[1] != ic:
                raise ValueError(f"Channel mismatch: expected {ic}, got {x.shape[1]}")

        if self.split_block == 1:
            return self._process_segment(inputs, 0)
        else:
            T = inputs[0].shape[2]
            seg_len = T // self.split_block

            segmented_inputs = []
            segment_shapes = []

            for x in inputs:
                sizes = [seg_len] * self.split_block
                sizes[-1] += x.shape[2] % self.split_block
                segment_shapes.append(sizes)
                segmented_inputs.append(torch.split(x, sizes, dim=2))

            processed_segments = []
            for t in range(self.split_block):
                seg_batch = [mod_segs[t] for mod_segs in segmented_inputs]
                # Use corresponding weights (t)
                out_seg = self._process_segment(seg_batch, t)
                processed_segments.append(out_seg)

            final_outputs = []
            for m in range(self.num_modality):
                mod_out = torch.cat([step[m] for step in processed_segments], dim=2)
                final_outputs.append(mod_out)

            return final_outputs


class HPFA(nn.Module):
    def __init__(self, args):
        super(HPFA, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_p = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert',
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'],
            save_hidden=False,
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'],
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'],
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'],
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'],
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])

        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'],
                        save_hidden=False,
                        token_len=args['model']['dmc']['completeness_check']['token_length'],
                        dim=args['model']['dmc']['completeness_check']['input_dim'],
                        depth=args['model']['dmc']['completeness_check']['depth'],
                        heads=args['model']['dmc']['completeness_check']['heads'],
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'],
                          int(args['model']['dmc']['completeness_check']['hidden_dim'] / 2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim'] / 2), 1),
                nn.Sigmoid()),
        ])

        self.CCR = nn.ModuleList([
            nn.Sequential(
                Transformer(num_frames=args['model']['CCR']['input_length'],
                            save_hidden=False,
                            token_len=None,
                            dim=args['model']['CCR']['input_dim'],
                            depth=args['model']['CCR']['depth'],
                            heads=args['model']['CCR']['heads'],
                            mlp_dim=args['model']['CCR']['hidden_dim']),
                Sun(8, 8)  # Uses custom SunLaye r
            ) for _ in range(3)
        ])

        self.dmml = nn.ModuleList([
            Transformer(num_frames=args['model']['dmml']['language_encoder']['input_length'],
                        save_hidden=True,
                        token_len=None,
                        dim=args['model']['dmml']['language_encoder']['input_dim'],
                        depth=args['model']['dmml']['language_encoder']['depth'],
                        heads=args['model']['dmml']['language_encoder']['heads'],
                        mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim']),

            HhyperLearningEncoder(dim=args['model']['dmml']['hyper_modality_learning']['input_dim'],
                                  dim_head=int(args['model']['dmml']['hyper_modality_learning']['input_dim'] /
                                               args['model']['dmml']['hyper_modality_learning']['heads']),
                                  depth=args['model']['dmml']['hyper_modality_learning']['depth'],
                                  heads=args['model']['dmml']['hyper_modality_learning']['heads']),

            CrossTransformer(source_num_frames=args['model']['dmml']['fuison_transformer']['source_length'],
                             tgt_num_frames=args['model']['dmml']['fuison_transformer']['tgt_length'],
                             dim=args['model']['dmml']['fuison_transformer']['input_dim'],
                             depth=args['model']['dmml']['fuison_transformer']['depth'],
                             heads=args['model']['dmml']['fuison_transformer']['heads'],
                             mlp_dim=args['model']['dmml']['fuison_transformer']['hidden_dim']),

            nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])
        ])

        self.dual_interaction = DualPathInteraction(
            token_dim=8,
            channel_dim=128,
            hidden_dim=64,
            depth=2  # Default from original code
        )

        self.seg_attn_la = TemporalSegmentAttention(
            in_channels=[8, 8],
            block_channel=8,
            block_dropout=0.3,
            reduction_factor=2,
            split_block=2
        )

        self.seg_attn_lv = TemporalSegmentAttention(
            in_channels=[8, 8],
            block_channel=8,
            block_dropout=0.1,
            reduction_factor=1,
            split_block=2
        )

        self.seg_attn_ll = TemporalSegmentAttention(
            in_channels=[8, 8],
            block_channel=8,
            block_dropout=0.05,
            reduction_factor=2,
            split_block=2
        )

        self.fusion_weight_layer = nn.Sequential(
            nn.Linear(24, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)


        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        n_v = h_1_v
        n_a = h_1_a
        n_l = h_1_l

        m1 = [n_l, n_a]
        ff = self.seg_attn_la(m1)
        ll = ff[0]
        h_1_a = ff[1]

        m2 = [n_l, n_v]
        gg = self.seg_attn_lv(m2)
        LL = gg[0]
        h_1_v = gg[1]

        fusion_input = torch.cat([ll, LL, n_l], dim=-1)

        weights = self.fusion_weight_layer(fusion_input)

        alpha = weights[:, :, 0].unsqueeze(-1)
        beta = weights[:, :, 1].unsqueeze(-1)

        h_1_l = alpha * ll + beta * LL


        h_1_v = self.dual_interaction(h_1_v)
        h_1_a = self.dual_interaction(h_1_a)
        h_1_l = self.dual_interaction(h_1_l)


        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp)

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b=b)
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)

        h_1_d = h_1_p * (1 - w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)


        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        h_d_list = self.dmml[0](h_1_d)
        h_hyper = self.dmml[1](h_d_list, h_1_a, h_1_v, h_hyper)
        feat = self.dmml[2](h_hyper, h_d_list[-1])

        output = self.dmml[3](torch.mean(feat[:, 1:], dim=1))


        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            rec_feat_a = self.CCR[0](h_1_a)
            rec_feat_v = self.CCR[1](h_1_v)
            rec_feat_l = self.CCR[2](h_1_l)

            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            effective_discriminator_input = rearrange(torch.cat([h_1_d, complete_language_feat]), 'b n d -> (b n) d')
            effectiveness_discriminator_out = self.effective_discriminator(effective_discriminator_input)

            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1)

        return {'sentiment_preds': output,
                'joint_representation': h_1_d,
                'w': w,
                'effectiveness_discriminator_out': effectiveness_discriminator_out,
                'rec_feats': rec_feats,
                'complete_feats': complete_feats}


def build_model(args):
    return HPFA(args)