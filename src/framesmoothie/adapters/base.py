from __future__ import annotations

import abc
import torch
import torch.nn as nn


class AdapterHubBase(nn.Module, abc.ABC):
    """Base class for adapter hubs."""

    @abc.abstractmethod
    def set_context(self, ctx: torch.Tensor) -> None:
        """ctx: [B, ctx_dim]"""
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class ModuleAdapterBase(nn.Module, abc.ABC):
    """Adapter plugin interface (swappable)."""

    hub: AdapterHubBase

    @abc.abstractmethod
    def set_context(self, ctx: torch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_linear(self, base: nn.Linear) -> nn.Module:
        raise NotImplementedError

    # Optional hooks for future extensions.
    def wrap_conv1x1(self, base: nn.Module) -> nn.Module:
        return base
