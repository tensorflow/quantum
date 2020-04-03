# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A global singleton object for defining op execution parameters."""
import pathos


class QContext:
    """Class for storing quantum execution information."""

    def __init__(self):
        """Create quantum context."""

        # Currently unused property.
        # Will control whether batch_util.py or engine_util.py will be hit.
        self._engine_mode = False

    def _get_engine_mode(self):
        return self._engine_mode

    def _set_engine_mode(self, mode):
        self._engine_mode = mode


_Q_CONTEXT = None
_Q_CONTEXT_LOCK = pathos.helpers.mp.Lock()


def _set_context(ctx):
    global _Q_CONTEXT
    _Q_CONTEXT = ctx


def _create_context_safe():
    with _Q_CONTEXT_LOCK:
        # Acquired lock. Need to double check _Q_CONTEXT before creating.
        if _Q_CONTEXT is None:
            ctx = QContext()
            _set_context(ctx)


def q_context():
    """Get global quantum execution context."""
    if _Q_CONTEXT is None:
        _create_context_safe()
    return _Q_CONTEXT


def set_engine_mode(mode):
    """Set global engine mode in execution context."""
    q_context()._set_engine_mode(mode)


def get_engine_mode():
    """Get global engine mode in execution context."""
    return q_context()._get_engine_mode()
