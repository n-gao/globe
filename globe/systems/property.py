import inspect
import numbers
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from globe.utils import ema_make, ema_update, ema_value


class MolProperty:
    """
    Base class for molecular properties.
    """
    @property
    def value(self) -> numbers.Number:
        """
        Returns the value of the property.
        """
        raise NotImplementedError()
    
    @property
    def key(self) -> str:
        """
        Returns the key of the property.
        """
        raise NotImplementedError()

    def update(self, **kwargs) -> jax.Array:
        """
        Updates the property with given keyword arguments.

        Args:
            **kwargs: Keyword arguments to update the property.

        Returns:
            The updated property.
        """
        params = inspect.signature(self._update).parameters
        if any(k not in kwargs for k in params.keys()):
            return None
        return self._update(**{
            k: v for k, v in kwargs.items()
            if k in params
        })
    
    def _update(self) -> numbers.Number:
        """
        Updates the property.

        Returns:
            Value of the updated property.
        """
        raise NotImplementedError()


class WidthScheduler(MolProperty):
    """
    Class for width scheduler property.
    """
    width: jax.Array
    pmoves: np.ndarray
    target: float
    update_interval: int
    i: int = 0
    error: float = 0.025

    def __init__(self, init_width: jax.Array, target_pmove: float = 0.5, update_interval: int = 20):
        """
        Args:
            init_width: Initial width.
            target_pmove: Target probability move.
            update_interval: Update interval.
        """
        self.width = jnp.array(init_width, dtype=jnp.float32)
        self.target = target_pmove
        self.update_interval = update_interval
        self.pmoves = np.zeros((self.update_interval,))

    def _update(self, pmove: float) -> numbers.Number:
        """
        Updates the width scheduler property.

        Args:
            pmove: Probability move.

        Returns:
            The updated width scheduler property.
        """
        if self.i % self.update_interval == 0 and self.i > 0:
            pm_mean = self.pmoves.mean()
            if pm_mean < self.target - self.error:
                self.width /= 1.1
            elif pm_mean > self.target + self.error:
                self.width *= 1.1
        self.pmoves[self.i % self.update_interval] = pmove
        self.i += 1
        return self.width
    
    @property
    def value(self) -> numbers.Number:
        return self.width
    
    @property
    def key(self) -> str:
        return 'mcmc_width'


class EnergyStdEMA(MolProperty):
    """
    Class for energy standard EMA property.
    """
    decay: float
    ema: Tuple[jax.Array, jax.Array]
    
    def __init__(self, decay: float = 0.99):
        """
        Args:
            decay: Decay value.
        """
        self.decay = decay
        self.ema = ema_update(ema_make(jnp.ones(())), jnp.ones(()), self.decay)
    
    def _update(self, E_std: jax.Array) -> None:
        """
        Args:
            E_std: Energy standard deviation.
        """
        self.ema = ema_update(self.ema, E_std, self.decay)

    @property
    def value(self) -> numbers.Number:
        return ema_value(self.ema)
    
    @property
    def key(self) -> str:
        return 'std_ema'


class EnergyEMA(MolProperty):
    """
    Class for energy EMA property.
    """
    decay: float
    ema: Tuple[jax.Array, jax.Array]
    
    def __init__(self, decay: float = 0.99):
        """
        Args:
            decay: Decay value.
        """
        self.decay = decay
        self.ema = ema_make((jnp.zeros(()), jnp.zeros(())))
    
    def _update(self, E: jax.Array, E_err: jax.Array) -> None:
        """
        Args:
            E: Energy.
            E_err: Energy error.
        """
        self.ema = ema_update(self.ema, (E, E_err), self.decay)

    @property
    def value(self) -> numbers.Number:
        return ema_value(self.ema)
    
    @property
    def key(self) -> str:
        return 'energy'
