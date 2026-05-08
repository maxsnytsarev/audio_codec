import numpy as np
import torch
import torch.nn as nn


def k_means(Y, k, max_iters=100):
    # if Y.shape[0] < k:
    #     raise ValueError('Y is smaller than k')
    indeces = torch.randint(0, Y.shape[0], size=(k,), device=Y.device)
    ans = Y[indeces].clone()
    for _ in range(max_iters):
        objective = torch.sqrt((Y[:, None, :] - ans[None, :, :]).pow(2).sum(dim=-1))
        indeces = torch.argmin(objective, dim=1)
        new_ans = ans.clone()
        for j in range(k):
            mask = indeces == j
            if mask.any():
                new_ans[j] = Y[mask].mean(dim=0)

        if torch.allclose(new_ans, ans):
            break
        ans = new_ans

    return ans, indeces


class RQV(nn.Module):
    def __init__(self, N_q, D):
        super().__init__()
        self.N_q = N_q
        self.r = 75
        self.N = 2 ** (75 // N_q + 1)
        self.quntizers = nn.ModuleList()
        self.D = D
        for i in range(N_q):
            emb = nn.Embedding(self.N, D)
            emb.weight.requires_grad_(False)
            self.quntizers.append(emb)
        self.register_buffer("N_i", torch.zeros((N_q, self.N)))
        self.register_buffer("m_i", torch.zeros((N_q, self.N, D)))
        self.gamma = 0.99
        self.register_buffer("init", torch.tensor(False))

    def find(self, y, i):
        objective = (
            y.pow(2).sum(dim=-1)[..., None]
            + self.quntizers[i].weight.pow(2).sum(dim=-1)[None, None, ...]
            - 2 * y @ self.quntizers[i].weight.T
        )
        indeces = torch.argmin(objective, dim=-1)
        return self.quntizers[i](indeces), indeces

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        y = data_object.clone()
        b, d, s = y.shape
        y = y.transpose(2, 1)
        cur_y = torch.zeros_like(y)
        residual = y.clone()
        cur_count = torch.zeros_like(self.N_i)
        vec_count = torch.zeros_like(self.m_i)
        all_indeces = []
        with torch.no_grad():
            need_init = not self.init.item()

            for i in range(self.N_q):
                cur_res = residual.clone()
                if need_init:
                    flat_res = cur_res.reshape(-1, self.D)
                    centers, _ = k_means(flat_res, self.N)
                    self.quntizers[i].weight.copy_(centers)
                Q_i, indeces = self.find(cur_res, i)
                all_indeces.append(indeces)
                cur_y += Q_i
                for j in range(self.N):
                    cur_count[i, j] = torch.sum(indeces == j)
                    if cur_count[i, j] > 0:
                        vec_count[i, j] = torch.sum(cur_res[indeces == j], dim=0)
                    else:
                        vec_count[i, j] = torch.zeros(self.D, device=cur_res.device)
                    if need_init:
                        self.N_i[i, j] = cur_count[i, j]
                        self.m_i[i, j] = vec_count[i, j]
                    else:
                        self.N_i[i, j] = self.N_i[i, j] * self.gamma + cur_count[
                            i, j
                        ] * (1 - self.gamma)
                        self.m_i[i, j] = self.m_i[i, j] * self.gamma + vec_count[
                            i, j
                        ] * (1 - self.gamma)
                new_weight = self.m_i[i] / self.N_i[i].clamp(min=1e-8)[..., None]
                dead = self.N_i[i] < 2
                if need_init:
                    empty = self.N_i[i] == 0
                    new_weight[empty] = self.quntizers[i].weight[empty]
                else:
                    if dead.any():
                        flat_res = cur_res.reshape(-1, self.D)
                        random_ids = torch.randint(
                            0,
                            flat_res.shape[0],
                            size=(dead.sum().item(),),
                            device=cur_res.device,
                        )
                        new_weight[dead] = flat_res[random_ids]
                self.quntizers[i].weight.copy_(new_weight)
                residual -= Q_i
            if need_init:
                self.init.fill_(True)

        y = y.transpose(2, 1)
        cur_y = cur_y.transpose(2, 1)
        commitment_loss = torch.mean((cur_y.detach() - y) ** 2)
        all_indeces = torch.stack(all_indeces, dim=0)
        return {
            "logits": y + (cur_y - y).detach(),
            "commitment_loss": commitment_loss,
            "indeces": all_indeces,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
