# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

# 这个makefile负责执行一系列的python程序，包括导入图片、训练模型，将h5文件转换为tflite文件

MODEL_TRAIN = model/train.py
MODEL_CONVERT = model/h5_to_tflite.py
MODEL_H5 = model/$(MODEL_PREFIX).h5
# Increase this to improve accuracy
TRAINING_EPOCHS?=5

$(IMAGES):
# 在这里设置输入图片，放在images文件夹下
	echo "GENERATING INPUT IMAGES"
	(mkdir -p $(IMAGES); $(MODEL_PYTHON) model/save_samples.py -d $@ -n 5)

$(MODEL_H5): 
	$(MODEL_PYTHON) $(MODEL_TRAIN) $@ -e $(TRAINING_EPOCHS)

$(TRAINED_TFLITE_MODEL): $(MODEL_H5) | $(IMAGES)
	$(MODEL_PYTHON) $(MODEL_CONVERT) $< $@
