# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
if os.environ.get('FLINT_STEP0_ENABLE') == '1':
    try:
        from step0_audit import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(91)
