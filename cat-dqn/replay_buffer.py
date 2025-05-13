import torch
from torchrl.data import TensorDictPrioritizedReplayBuffer, PrioritizedSampler, LazyTensorStorage

class SequenceTensorDictPrioritizedReplayBuffer(TensorDictPrioritizedReplayBuffer):
    def __init__(self, storage, alpha=0.6, beta=0.4, sequence_length=5):
        self.alpha = alpha
        self.beta = beta

        super().__init__(storage=storage, alpha=alpha, beta=beta, )
        self.sequence_length = sequence_length

    def sample(self, batch_size, sequence_length=None, return_info=True):
        sequence_length = sequence_length or self.sequence_length

        # サンプリング時の優先度に基づくインデックスを取得
        indices, info = self._sampler.sample(self._storage, batch_size)
        info['index'] = indices

        # シーケンスの開始インデックスを計算
        start_indices = indices - (sequence_length - 1)
        start_indices = start_indices.clamp(min=0)

        # シーケンスの収集をループからバッチ処理に変更
        indices = torch.arange(sequence_length).unsqueeze(0) + start_indices.unsqueeze(1)  # [batch_size, sequence_length]

        batch_data = self._storage.get(indices.flatten())
        batch_data = batch_data.view(batch_size, sequence_length, *batch_data.shape[1:])
    
        return batch_data, info

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.view(len(indices), -1).max(dim=1)[0]
        # update_prioritiesで直接更新
        self._sampler.update_priorities(indices, td_errors.detach().cpu().numpy())
