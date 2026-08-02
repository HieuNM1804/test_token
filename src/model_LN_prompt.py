import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
)

from src.clip import clip
from experiments.options import opts

def freeze_all_but_layer_norm(module):
    """Freeze an encoder, then enable only its LayerNorm parameters."""
    module.requires_grad_(False)
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            child.requires_grad_(True)

class Model(pl.LightningModule):
    def __init__(self, categories):
        super().__init__()

        self.opts = opts
        self.categories = sorted(categories)
        self.category_to_index = {
            category: index for index, category in enumerate(self.categories)
        }
        self.save_hyperparameters({'categories': self.categories})

        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.sketch_visual = copy.deepcopy(self.clip.visual)

        self.clip.requires_grad_(False)
        freeze_all_but_layer_norm(self.clip.visual)
        freeze_all_but_layer_norm(self.sketch_visual)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        text_prompts = [
            'a photo of a %s' % category.replace('_', ' ')
            for category in self.categories
        ]
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
        self.register_buffer('text_features', text_features)

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.margin)

        self._query_outputs = []
        self._gallery_outputs = []

    def configure_optimizers(self):
        visual_parameters = [
            parameter
            for encoder in (self.clip.visual, self.sketch_visual)
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam([
            {'params': visual_parameters, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt, self.img_prompt], 'lr': self.opts.prompt_lr}
        ])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.visual(
                data.type(self.clip.dtype),
                self.img_prompt.expand(data.shape[0], -1, -1).type(self.clip.dtype))
        else:
            feat = self.sketch_visual(
                data.type(self.clip.dtype),
                self.sk_prompt.expand(data.shape[0], -1, -1).type(self.clip.dtype))
        return feat

    def classification_loss(self, features, categories):
        targets = torch.tensor(
            [self.category_to_index[category] for category in categories],
            dtype=torch.long,
            device=features.device)
        normalized_features = F.normalize(features.float(), dim=-1)
        logits = self.clip.logit_scale.exp().detach().float()
        logits = logits * normalized_features @ self.text_features.t()
        return F.cross_entropy(logits, targets)

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        sketch_classification_loss = self.classification_loss(sk_feat, category)
        photo_classification_loss = self.classification_loss(img_feat, category)
        loss = (
            self.opts.triplet_weight * triplet_loss
            + self.opts.classification_weight * (
                sketch_classification_loss + photo_classification_loss)
        )

        batch_size = sk_tensor.shape[0]
        self.log(
            'train_loss', loss, on_step=False, on_epoch=True,
            prog_bar=False, batch_size=batch_size)
        self.log(
            'train_triplet_loss', triplet_loss, on_step=False, on_epoch=True,
            batch_size=batch_size)
        self.log(
            'train_sketch_cls_loss', sketch_classification_loss,
            on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            'train_photo_cls_loss', photo_classification_loss,
            on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self._query_outputs = []
        self._gallery_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, category, sample_index = batch
        if dataloader_idx == 0:
            features = self.forward(image, dtype='sketch')
            self._query_outputs.append(
                (features.detach(), category.detach(), sample_index.detach()))
        else:
            features = self.forward(image, dtype='image')
            self._gallery_outputs.append(
                (features.detach(), category.detach(), sample_index.detach()))

    def _gather_unique_outputs(self, outputs):
        features = torch.cat([output[0] for output in outputs], dim=0)
        categories = torch.cat([output[1] for output in outputs], dim=0)
        sample_indices = torch.cat([output[2] for output in outputs], dim=0)

        features = self.all_gather(features, sync_grads=False)
        categories = self.all_gather(categories, sync_grads=False)
        sample_indices = self.all_gather(sample_indices, sync_grads=False)

        if features.ndim == 3:
            features = features.flatten(0, 1)
            categories = categories.flatten(0, 1)
            sample_indices = sample_indices.flatten(0, 1)

        # DistributedSampler may pad a shard. Keep one copy per source file.
        order = torch.argsort(sample_indices)
        features = features[order]
        categories = categories[order]
        sample_indices = sample_indices[order]
        keep = torch.ones_like(sample_indices, dtype=torch.bool)
        keep[1:] = sample_indices[1:] != sample_indices[:-1]
        return features[keep], categories[keep]

    @staticmethod
    def retrieval_metrics_at_k(
            query_features, query_categories,
            gallery_features, gallery_categories, k=200):
        """Compute category-level mAP@K and P@K from rescaled cosine scores."""

        query_features = query_features.float()
        gallery_features = gallery_features.float()
        query_categories = query_categories.cpu()
        gallery_categories = gallery_categories.cpu()

        average_precision = torch.zeros(len(query_features))
        precision_at_k = torch.zeros(len(query_features))
        map_k = min(k, len(gallery_features))

        for index, query_feature in enumerate(query_features):
            cosine_similarities = F.cosine_similarity(
                query_feature.unsqueeze(0), gallery_features
            ).cpu()
            # TorchMetrics 1.8+ treats scores <= 0 as absent predictions.
            # Preserve cosine ranking while mapping [-1, 1] to [0, 1].
            similarities = (cosine_similarities + 1.0) / 2.0
            target = gallery_categories.eq(query_categories[index])
            average_precision[index] = retrieval_average_precision(
                similarities, target, top_k=map_k)
            precision_at_k[index] = retrieval_precision(
                similarities, target, top_k=k)

        return average_precision.mean(), precision_at_k.mean()

    def on_validation_epoch_end(self):
        if not self._query_outputs or not self._gallery_outputs:
            return

        query_features, query_categories = self._gather_unique_outputs(
            self._query_outputs)
        gallery_features, gallery_categories = self._gather_unique_outputs(
            self._gallery_outputs)
        mean_average_precision, precision = self.retrieval_metrics_at_k(
            query_features,
            query_categories,
            gallery_features,
            gallery_categories,
            k=200)

        self.log('mAP200', mean_average_precision, prog_bar=False)
        self.log('P200', precision, prog_bar=False)

        if self.trainer.is_global_zero:
            print(
                'Epoch {:03d} | mAP@200: {:.6f} | P@200: {:.6f}'.format(
                    self.current_epoch + 1,
                    mean_average_precision.item(),
                    precision.item()))
