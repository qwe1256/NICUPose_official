# Installation

Opera relies on several basic packages such as MMCV, MMDetection, etc, so you need to install these packages at first.

1. Install with `mim`

   ```bash
   pip install -U openmim
   ```
2. Install `mmcv`

   ```bash

   mim install mmcv-full==1.7.0
   ```
3. Install `mmdet`

   ```bash
   mim install mmdet==2.27.0
   ```
4. Install `NICUPose`

   ```bash
   cd /ROOT/
   pip install -r requirements.txt
   pip install -e .
   ```
