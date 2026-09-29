# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Unit tests for `fold_cp_ops._internal.port_selection`.

Subject: the rule that replaced two baked port bands. The load-bearing property is not "returns a
number" but **"never returns an ephemeral-range port, and never returns a port that is taken"** --
the two ways the bands it replaced were wrong. Both are asserted against a real bound socket
rather than a mock, because the defect being prevented is a real `bind` failing on a real host.

No GPU, no process group, no distributed anything -- this is a socket and a sysctl file.
"""

from __future__ import annotations

import socket

import pytest

from fold_cp_ops._internal.port_selection import (
    ephemeral_floor,
    is_port_free,
)


@pytest.fixture
def bound_port():
    """Bind a real socket and yield its port, so "taken" means taken rather than mocked.

    Yields:
        An int port held by a live listening socket for the duration of the test. The socket is
        closed on teardown. `SO_REUSEADDR` is not set, matching `is_port_free`'s probe, so the
        test and the code under test disagree about nothing.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


def test_a_taken_port_is_reported_taken(bound_port):
    """`is_port_free` must return False for a port a live socket is holding.

    This is the check the arithmetic bands could not perform at all, and the reason the failure
    they caused (`EADDRINUSE` at 42176) was invisible until a cell burned its timeout.
    """
    assert is_port_free(bound_port) is False


def test_the_floor_comes_from_the_file_not_from_a_constant(tmp_path):
    """`ephemeral_floor` parses the sysctl file, which is what makes the rule portable.

    The point of the module is that no port number is written in it. If the floor were a constant,
    this whole design would be a second baked band with better prose.
    """
    f = tmp_path / "range"
    f.write_text("40000\t60999\n")
    assert ephemeral_floor(str(f)) == 40000
