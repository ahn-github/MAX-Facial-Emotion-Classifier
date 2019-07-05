#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
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
#

import cv2
import numpy as np

def img_resize(input_data):
    # ratio = 1
    img_h, img_w, _ = np.shape(input_data)
    if img_w > 1024:
        ratio = 1024/img_w
        input_data = cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    elif img_h > 1024:
        ratio = 1024/img_h
        input_data = cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    return input_data #, ratio
