import itertools
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F

from ..utils.lddmm_params import *


class TransformParamsLayer(nn.Module):
    def __init__(self):
        super(TransformParamsLayer, self).__init__()

    def forward(self, x, mean_shape):
        batch_size = x.size(0)
        dest = mean_shape
        source = x.view(batch_size, -1, 2)
    
        dest_mean = torch.mean(dest, dim=1, keepdim=True)
        src_mean = torch.mean(source, dim=1, keepdim=True)

        dest_vec = (dest - dest_mean).view(batch_size, -1, 1)
        src_vec = (source - src_mean).view(batch_size, 1, -1)

        norm = torch.linalg.norm(src_vec, ord=2, dim=-1)**2
        # (b, 1, c) @ (b, c, 1) -> (b, 1, 1)
        a = torch.bmm(src_vec, dest_vec).squeeze() / norm.squeeze()
        b = torch.zeros((batch_size, 1)).cuda()
        for i in range(mean_shape.size(1)):
            b[:, 0] += (src_vec[..., 2*i] * dest_vec[:, 2*i+1]).squeeze() - \
                       (src_vec[..., 2*i+1] * dest_vec[:, 2*i]).squeeze()
        b /= norm

        A = torch.zeros((batch_size, 2, 2)).cuda()
        A[:, 0, 0] = a.squeeze()
        A[:, 0, 1] = b.squeeze()
        A[:, 1, 0] = -b.squeeze()
        A[:, 1, 1] = a.squeeze()
        # (b, 1, 2) @ (b, 2, 2) -> (b, 1, 2)
        src_mean = torch.bmm(src_mean, A)

        return torch.cat([A.view(batch_size, 4), dest_mean.squeeze(1) - src_mean.squeeze(1)], dim=1)


class AffineTransformLayer(nn.Module):
    def __init__(self):
        super(AffineTransformLayer, self).__init__()

    def affine_transform(self, x, A, t):
        img = x

        pixels = torch.tensor([(x_, y) for x_ in range(self.w) for y in range(self.h)], 
                                                dtype=torch.float32).cuda()
        pixels = torch.cat(self.batch_size*[pixels.view(1, -1, 2)], dim=0)
        # (b, h*w, 2) @ (b, 2, 2) + (b, 1, 2)
        output_pixels = torch.bmm(pixels, A) + t
        clipped_output_pixels = torch.zeros_like(output_pixels)

        clipped_output_pixels[..., 0] = torch.clamp(output_pixels[..., 0], min=0, max=self.h - 2)
        clipped_output_pixels[..., 1] = torch.clamp(output_pixels[..., 1], min=0, max=self.w - 2)

        output_pixels_min_min = clipped_output_pixels.long()
        output_pixels_max_min = output_pixels_min_min + torch.tensor([1, 0]).cuda()
        output_pixels_min_max = output_pixels_min_min + torch.tensor([0, 1]).cuda()
        output_pixels_max_max = output_pixels_min_min + torch.tensor([1, 1]).cuda()

        dx = clipped_output_pixels[..., 0] - output_pixels_min_min[..., 0]
        dy = clipped_output_pixels[..., 1] - output_pixels_min_min[..., 1]

        pixels = pixels.long()

        output_img = torch.zeros((self.batch_size, img.size(1), self.h, self.w)).cuda()
        for i in range(self.batch_size):
            output_img[i, :, pixels[0, :, 1], pixels[0, :, 0]] += (1 - dx[i, 0]) * (1 - dy[i, 0]) * img[i, :, output_pixels_min_min[i, :, 1], output_pixels_min_min[i, :, 0]]
            output_img[i, :, pixels[0, :, 1], pixels[0, :, 0]] += dx[i, 0] * (1 - dy[i, 0]) * img[i, :, output_pixels_max_min[i, :, 1], output_pixels_max_min[i, :, 0]]
            output_img[i, :, pixels[0, :, 1], pixels[0, :, 0]] += (1 - dx[i, 0]) * dy[i, 0] * img[i, :, output_pixels_min_max[i, :, 1], output_pixels_min_max[i, :, 0]]
            output_img[i, :, pixels[0, :, 1], pixels[0, :, 0]] += dx[i, 0] * dy[i, 0] * img[i, :, output_pixels_max_max[i, :, 1], output_pixels_max_max[i, :, 0]]

        return output_img

    def forward(self, x, transform):
        self.batch_size = x.size(0)
        self.w = self.h = x.size(-1)
        A = torch.zeros((self.batch_size, 2, 2)).cuda()

        A[:, 0, 0] = transform[:, 0]
        A[:, 0, 1] = transform[:, 1]
        A[:, 1, 0] = transform[:, 2]
        A[:, 1, 1] = transform[:, 3]
        t = transform[:, 4:].view(self.batch_size, 1, 2)

        A = torch.inverse(A)
        # (b, 1, 2) @ (b, 2, 2)
        t = torch.bmm(-t, A)
        theta = torch.cat([A, t.view(-1, 2, 1)], dim=-1)

        return self.affine_transform(x, A, t)


class LandmarkTransformLayer(nn.Module):
    def __init__(self):
        super(LandmarkTransformLayer, self).__init__()

    def forward(self, landmarks, transform, inverse=False):
        batch_size = landmarks.size(0)
        A = torch.zeros([batch_size, 2, 2]).cuda()
        A[:, 0, 0] = transform[:, 0].clone()
        A[:, 0, 1] = transform[:, 1].clone()
        A[:, 1, 0] = transform[:, 2].clone()
        A[:, 1, 1] = transform[:, 3].clone()
        t = transform[:, 4:].view(batch_size, 1, 2).clone()

        if inverse:
            A = torch.inverse(A)
            # (b, 1, 2) @ (b, 2, 2)
            t = torch.bmm(-t, A)

        # (b, c, 2) @ (b, 2, 2) + (b, 1, 2)
        output = torch.bmm(landmarks.view(batch_size, -1, 2), A) + t
        
        return output.view(batch_size, -1)


class LandmarkImageLayer(nn.Module):
    def __init__(self, patch_size):
        super(LandmarkImageLayer, self).__init__()
        self.patch_size = patch_size
        self.half_size = int(patch_size / 2)
        self.offsets = torch.tensor(list(itertools.product(range(-self.half_size, self.half_size + 1),
                                    range(-self.half_size, self.half_size + 1)))).cuda()

    def draw_landmark(self, landmark):      
        landmark = landmark.unsqueeze(1)

        landmark_long = landmark.long()
        offsets_b = torch.cat(landmark.size(-2)*[self.offsets.unsqueeze(1)], dim=1).long()
        offsets_b = torch.cat(self.batch_size*[offsets_b.unsqueeze(0)], dim=0).long()
        # (b, 1, 68, 2) + (b, c', 68, 2) -> (b, c', 68, 2)
        locations = offsets_b + landmark_long
        # (b, c', 136)
        flatten_locations = (locations[..., 0] + locations[..., 1] * self.w).view(self.batch_size, 1, -1)
        dxdy = landmark - landmark_long

        offsets_sub_pix = offsets_b - dxdy
        vals = 1 / (1 + torch.sqrt(torch.sum(offsets_sub_pix**2, dim=-1) + 1e-6))
        vals = vals.view(self.batch_size, 1, -1)
        self.img.view(self.batch_size, 1, -1).scatter_add_(-1, flatten_locations, vals)
        self.img = self.img.view(self.batch_size, 1, self.h, self.w)

    def forward(self, img, landmarks):
        self.batch_size = landmarks.size(0)
        self.h, self.w = img.size(2), img.size(3) 
        self.offsets_b = torch.cat([self.offsets.unsqueeze(0)]*self.batch_size, dim=0)     
        landmarks = landmarks.view(self.batch_size, -1, 2)
        landmarks = torch.clamp(landmarks, self.half_size, self.h - 1 - self.half_size)

        self.img = torch.zeros((self.batch_size, 1, self.h, self.w)).cuda()
        self.draw_landmark(landmarks)

        return self.img


class LandmarkHeatmapLayer(nn.Module):
    def __init__(self, patch_size):
        super(LandmarkImageLayer, self).__init__()
        self.patch_size = patch_size
        self.half_size = int(patch_size / 2)

    def draw_landmark(self, landmark):      
        landmark = landmark.unsqueeze(1)

        landmark_long = landmark.long()
        offsets_b = torch.cat(landmark.size(-2)*[self.offsets.unsqueeze(1)], dim=1).long()
        offsets_b = torch.cat(self.batch_size*[offsets_b.unsqueeze(0)], dim=0).long()
        # (b, 1, 68, 2) + (b, c', 68, 2) -> (b, c', 68, 2)
        locations = offsets_b + landmark_long
        # (b, c', 136)
        flatten_locations = (locations[..., 0] + locations[..., 1] * self.w).view(self.batch_size, 1, -1)
        dxdy = landmark - landmark_long

        offsets_sub_pix = offsets_b - dxdy
        vals = 1 / (1 + torch.sqrt(torch.sum(offsets_sub_pix**2, dim=-1) + 1e-6))
        vals = vals.view(self.batch_size, 1, -1)
        self.img.view(self.batch_size, 1, -1).scatter_add_(-1, flatten_locations, vals)
        self.img = self.img.view(self.batch_size, 1, self.h, self.w)

    def forward(self, img, landmarks):
        # TODO: n-layer heatmap
        self.batch_size = landmarks.size(0)
        self.h, self.w = img.size(2), img.size(3)  
        landmarks = landmarks.view(self.batch_size, -1, 2)
        landmarks = torch.clamp(landmarks, self.half_size, self.h - 1 - self.half_size)

        self.img = torch.zeros((self.batch_size, 1, self.h, self.w)).cuda()
        self.draw_landmark(landmarks)

        return self.img


class LandmarkDeformLayer(nn.Module):
    def __init__(self, config, n_landmark, n_T=3):
        super(LandmarkDeformLayer, self).__init__()

        self.n_T = n_T
        self.tau = 1 / (self.n_T - 1)
        self.n_landmark = n_landmark 
        self.sigmaV2 = get_sigmaV2(config.DATASET.DATASET, n_landmark)
        self.curve2landmark = get_curve2landmark(config.DATASET.DATASET, n_landmark)

        if config.TEST.INFERENCE and (config.MODEL.NUM_JOINTS != n_landmark):
            self.broadcast_index = get_broadcast_index(config.DATASET.DATASET, config.MODEL.NUM_JOINTS, n_landmark)
        else:
            self.broadcast_index = None

        self.mask = torch.zeros((1, n_landmark, n_landmark, 2)).cuda()
        for k, v in self.curve2landmark.items():
            for index, landmark in enumerate(list(v)):
                self.mask[0, landmark, v, :] = 1


    def forward(self, momentum, init_landmark):
        batch_size = momentum.size(0)
        n_pt = init_landmark.size(1)
        if self.broadcast_index:
            broadcast_momentum = torch.zeros_like(init_landmark)
            broadcast_momentum[:, self.broadcast_index, :] = momentum.view(batch_size, -1, 2)
            momentum = broadcast_momentum
        # guassian operator (b, n_landmark, 2)
        dp1 = torch.zeros_like(momentum)
        dp2 = torch.zeros_like(momentum)

        momentum = momentum.view(batch_size, 1, -1, 2)
        momentums = torch.cat([momentum]*self.n_landmark, dim=1)
        init_landmark = init_landmark.view(batch_size, 1, -1, 2)
        init_landmarks = torch.cat([init_landmark]*self.n_landmark, dim=1)

        # T = 1
        mask = torch.cat([self.mask]*batch_size, dim=0)
        masked_init_landmarks = init_landmarks * mask
        masked_init_landmarks_location = init_landmarks.permute(0, 2, 1, 3) * mask
        masked_momentums = momentums * mask
        sigmaV2 = torch.cat([self.sigmaV2.view(1, n_pt, 1)]*batch_size, dim=0)

        # (b, n_landmark, n_landmark)
        weight = torch.exp(-torch.sum(
                    (masked_init_landmarks_location - masked_init_landmarks) ** 2, 
                    dim=-1) / sigmaV2)
        # softmax weight
        # weight = torch.softmax(weight, dim=2)
        # (b, n_landmark, n_landmark, 2)
        weight = torch.cat([weight.view(batch_size, self.n_landmark, 
                                        self.n_landmark, 1)]*2, dim=-1)
        # (b, n_landmark, 2)
        dp1 = torch.sum(weight * masked_momentums, dim=-2)
        deformed_shape = init_landmark.squeeze(1) + dp1 * self.tau

        # T = 2
        deformed_shape = deformed_shape.view(batch_size, 1, -1, 2)
        deformed_shapes = torch.cat([deformed_shape]*self.n_landmark, dim=1)
        masked_deformed_shapes = deformed_shapes * mask
        masked_deformed_shapes_location = deformed_shapes.permute(0, 2, 1, 3) * mask

        # (b, n_landmark, n_landmark)
        weight = torch.exp(-torch.sum(
                    (masked_deformed_shapes_location - masked_deformed_shapes) ** 2, 
                    dim=-1) / sigmaV2)
        # (b, n_landmark, n_landmark, 2)
        weight = torch.cat([weight.view(batch_size, self.n_landmark, 
                                        self.n_landmark, 1)]*2, dim=-1)
        # (b, n_landmark, 2)
        dp2 = torch.sum(weight * masked_momentums, dim=-2)
        output = deformed_shape.squeeze(1) + dp2 * self.tau

        return output.view(batch_size, -1, 2)


class SingleCurveDeformLayer(nn.Module):
    def __init__(self, n_T=3):
        super(SingleCurveDeformLayer, self).__init__()

        self.n_T = n_T
        self.tau = 1 / (self.n_T - 1)

    def forward(self, momentum, source_init, target_init, sigmaV2):
        batch_size = momentum.size(0)
        n_momentum = momentum.size(1)
        n_pt = source_init.size(1) + target_init.size(1)
        init_landmark = torch.cat([source_init, target_init], dim=1)
        broadcast_momentum = torch.zeros_like(init_landmark)
        broadcast_momentum[:, 0:n_momentum] = momentum.view(batch_size, -1, 2)
        momentum = broadcast_momentum
        # guassian operator (b, n_landmark, 2)
        dp1 = torch.zeros_like(momentum)
        dp2 = torch.zeros_like(momentum)

        momentum = momentum.view(batch_size, 1, -1, 2)
        momentums = torch.cat([momentum]*n_pt, dim=1)
        init_landmark = init_landmark.view(batch_size, 1, -1, 2)
        init_landmarks = torch.cat([init_landmark]*n_pt, dim=1)

        # T = 1
        masked_init_landmarks = init_landmarks
        masked_init_landmarks_location = masked_init_landmarks.permute(0, 2, 1, 3)
        masked_momentums = momentums

        # (b, n_landmark, n_landmark)
        weight = torch.exp(-torch.sum(
                    (masked_init_landmarks_location - masked_init_landmarks) ** 2, 
                    dim=-1) / sigmaV2)
        # (b, n_landmark, n_landmark, 2)
        weight = torch.cat([weight.view(batch_size, n_pt, n_pt, 1)]*2, dim=-1)
        # (b, n_landmark, 2)
        dp1 = torch.sum(weight * masked_momentums, dim=-2)
        deformed_shape = init_landmark.squeeze(1) + dp1 * self.tau

        # T = 2
        deformed_shape = deformed_shape.view(batch_size, 1, -1, 2)
        deformed_shapes = torch.cat([deformed_shape]*n_pt, dim=1)
        masked_deformed_shapes = deformed_shapes
        masked_deformed_shapes_location = masked_deformed_shapes.permute(0, 2, 1, 3)

        # (b, n_landmark, n_landmark)
        weight = torch.exp(-torch.sum(
                    (masked_deformed_shapes_location - masked_deformed_shapes) ** 2, 
                    dim=-1) / sigmaV2)
        # (b, n_landmark, n_landmark, 2)
        weight = torch.cat([weight.view(batch_size, n_pt, n_pt, 1)]*2, dim=-1)
        # (b, n_landmark, 2)
        dp2 = torch.sum(weight * masked_momentums, dim=-2)
        output = deformed_shape.squeeze(1) + dp2 * self.tau

        return output.view(batch_size, -1, 2)[:, n_momentum:]
