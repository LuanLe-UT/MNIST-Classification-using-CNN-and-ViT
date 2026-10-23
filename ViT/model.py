"""Vision Transformer classifier for 28x28 grayscale MNIST images."""

import torch
from torch import nn

if __package__:
    from .layers import Dense, Dropout, EncoderBlock, LayerNorm, PositionalEncoding
else:
    from layers import Dense, Dropout, EncoderBlock, LayerNorm, PositionalEncoding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size=2,
        image_size=28,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        patches_per_side = image_size // patch_size
        num_patches = patches_per_side * patches_per_side
        patch_values = patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embedding = Dense(patch_values, embedding_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_encoding = PositionalEncoding(num_patches + 1, embedding_dim)
        self.embedding_dropout = Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim, num_heads, embedding_dim * 4, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = LayerNorm(embedding_dim)
        self.classifier_dropout = Dropout(dropout)
        self.classifier = Dense(embedding_dim, num_classes)

    def forward(self, patches):
        if patches.ndim != 3:
            raise ValueError("patches must have shape (batch, number_of_patches, values)")
        tokens = self.patch_embedding(patches)
        class_tokens = self.class_token.expand(patches.shape[0], -1, -1)
        tokens = self.position_encoding(
            torch.cat((class_tokens, tokens), dim=1)
        )
        tokens = self.embedding_dropout(tokens)
        for encoder_block in self.encoder_blocks:
            tokens = encoder_block(tokens)
        representation = self.output_norm(tokens[:, 0])
        return self.classifier(self.classifier_dropout(representation))
