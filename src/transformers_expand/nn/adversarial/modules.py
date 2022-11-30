# -*- coding: utf-8 -*-
# @Time     : 2022/11/19 15:54
# @File     : modules.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
import torch


class _Adversarial(object):
    """
    Base class for all Adversarial
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.args = None if trainer is None else trainer.args

    def __call__(self, inputs):
        return self.forward(inputs=inputs)

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, inputs):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def attack(self):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"attack\" function")

    def restore(self):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"restore\" function")

    def _get_args(self, args_name, no_found_return=None):
        if hasattr(self.args, args_name):
            return getattr(self.args, args_name)
        else:
            return no_found_return


class FGSM(_Adversarial):
    """
    FGSM 对抗训练方案

    """
    def __init__(self, trainer):
        super(FGSM, self).__init__(trainer=trainer)
        self.emb_name = self._get_args('FGSM_emb_name', 'embeddings')
        self.epsilon = self._get_args('FGSM_epsilon', 1.)
        self.backup = {}

    def __str__(self):
        return f"{self.__class__.__name__}(epsilon={self.epsilon})"

    def attack(self):
        for name, param in self.trainer.model.named_parameters():

            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.epsilon * param.grad.sign()
                param.data.add_(r_at)

    def restore(self):
        for name, para in self.trainer.model.named_parameters():
            if para.requires_grad and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

    def forward(self, inputs):
        self.attack()
        with self.trainer.compute_loss_context_manager():
            loss = self.trainer.compute_loss(self.trainer.model, inputs)
        loss.backward()
        self.restore()


class FGM(_Adversarial):
    """
    FGM 对抗训练方案

    """
    def __init__(self, trainer):
        super(FGM, self).__init__(trainer=trainer)
        self.emb_name = self._get_args('FGM_emb_name', 'embeddings')
        self.epsilon = self._get_args('FGM_epsilon', 1.)
        self.backup = {}

    def __str__(self):
        return f"{self.__class__.__name__}(epsilon={self.epsilon})"

    def attack(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def forward(self, inputs):
        self.attack()
        with self.trainer.compute_loss_context_manager():
            loss = self.trainer.compute_loss(self.trainer.model, inputs)
        loss.backward()
        self.restore()


class PGD(_Adversarial):
    """
    PGD 对抗训练方案

    """
    def __init__(self, trainer):
        super(PGD, self).__init__(trainer=trainer)
        self.emb_name = self._get_args('PGD_emb_name', 'embeddings')
        self.epsilon = self._get_args('PGD_epsilon', 1.)
        self.steps = self._get_args('PGD_steps', 3)
        self.alpha = self._get_args('PGD_alpha', .3)

        self.emb_backup = {}
        self.grad_backup = {}

    def __str__(self):
        return f"{self.__class__.__name__}(steps={self.steps}, epsilon={self.epsilon}, alpha={self.alpha})"

    def _project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def attack(self, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    print(r_at.shape, param.data.shape)
                    param.data.add_(r_at)
                    param.data = self._project(name, param.data, self.epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

    def forward(self, inputs):
        self.backup_grad()
        # 对抗训练
        for t in range(self.steps):
            self.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != self.steps - 1:
                self.trainer.model.zero_grad()
            else:
                self.restore_grad()
            with self.trainer.compute_loss_context_manager():
                loss = self.trainer.compute_loss(self.trainer.model, inputs)
            loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        self.restore()  # 恢复embedding参数


class FreeAT(_Adversarial):
    """
    FreeAT 对抗训练方案

    """
    def __init__(self, trainer):
        super(FreeAT, self).__init__(trainer=trainer)
        self.epsilon = self._get_args('FreeAT_epsilon', 1.)
        self.emb_name = self._get_args('FreeAT_emb_name', 'embeddings')
        self.steps = self._get_args('FreeAT_steps', 3)
        self.emb_backup = {}
        self.grad_backup = {}
        self.last_r_at = {}

    def __str__(self):
        return f"{self.__class__.__name__}(epsilon={self.epsilon}, steps={self.steps})"

    def attack(self, is_first_attack=False):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                param.data.add_(self.last_r_at.get(name, 0))
                param.data = self.project(name, param.data)
                self.last_r_at[name] = self.last_r_at.get(name, 0) + self.epsilon * param.grad.sign()

    def restore(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):

        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

    def forward(self, inputs):
        self.backup_grad()
        # 对抗训练
        for t in range(self.steps):
            self.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != self.steps - 1:
                self.trainer.model.zero_grad()
            else:
                self.restore_grad()
            with self.trainer.compute_loss_context_manager():
                loss = self.trainer.compute_loss(self.trainer.model, inputs)
            loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        self.restore()  # 恢复embedding参数
