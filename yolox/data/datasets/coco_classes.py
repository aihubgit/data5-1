#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

COCO_CLASSES = (
    #"person",
    #"bicycle",
    #"car",
    #"motorcycle",
    #"airplane",
    #"bus",
    #"train",
    #"truck",
    #"boat",
    #"traffic light",
    #"fire hydrant",
    #"stop sign",
    #"parking meter",
    #"bench",
    #"bird",
    #"cat",
    #"dog",
    #"horse",
    #"sheep",
    #"cow",
    #"elephant",
    #"bear",
    #"zebra",
    #"giraffe",
    #"backpack",
    #"umbrella",
    #"handbag",
    #"tie",
    #"suitcase",
    #"frisbee",
    #"skis",
    #"snowboard",
    #"sports ball",
    #"kite",
    #"baseball bat",
    #"baseball glove",
    #"skateboard",
    #"surfboard",
    #"tennis racket",
    #"bottle",
    #"wine glass",
    #"cup",
    #"fork",
    #"knife",
    #"spoon",
    #"bowl",
    #"banana",
    #"apple",
    #"sandwich",
    #"orange",
    #"broccoli",
    #"carrot",
    #"hot dog",
    #"pizza",
    #"donut",
    #"cake",
    #"chair",
    #"couch",
    #"potted plant",
    #"bed",
    #"dining table",
    #"toilet",
    #"tv",
    #"laptop",
    #"mouse",
    #"remote",
    #"keyboard",
    #"cell phone",
    #"microwave",
    #"oven",
    #"toaster",
    #"sink",
    #"refrigerator",
    #"book",
    #"clock",
    #"vase",
    #"scissors",
    #"teddy bear",
    #"hair drier",
    #"toothbrush",
    'B01_C01_D10', 
    'B01_C01_D20', 
    'B01_C01_D30', 
    'B01_C01_D40', 
    'B01_C01_D50', 
    'B01_C01_D60', 
    'B01_C02_D10', 
    'B01_C02_D20', 
    'B01_C02_D30', 
    'B01_C02_D40', 
    'B01_C02_D50', 
    'B01_C02_D60', 
    'B01_C03_D10', 
    'B01_C03_D20', 
    'B01_C03_D30', 
    'B01_C03_D40', 
    'B01_C03_D50', 
    'B01_C03_D60', 
    'B01_C04_D10', 
    'B01_C04_D20', 
    'B01_C04_D30', 
    'B01_C04_D40', 
    'B01_C04_D50', 
    'B01_C04_D60', 
    'B01_C05_D10', 
    'B01_C05_D20', 
    'B01_C05_D30', 
    'B01_C05_D40', 
    'B01_C05_D50', 
    'B01_C05_D60', 
    'B01_C06_D10', 
    'B01_C06_D20', 
    'B01_C06_D30', 
    'B01_C06_D40', 
    'B01_C06_D50', 
    'B01_C06_D60', 
    'B01_C07_D10', 
    'B01_C07_D20', 
    'B01_C07_D30', 
    'B01_C07_D40', 
    'B01_C07_D50', 
    'B01_C07_D60', 
    'B01_C08_D10', 
    'B01_C08_D20', 
    'B01_C08_D30', 
    'B01_C08_D40', 
    'B01_C08_D50', 
    'B01_C08_D60', 
    'B01_C09_D10', 
    'B01_C09_D20', 
    'B01_C09_D30', 
    'B01_C09_D40', 
    'B01_C09_D50', 
    'B01_C09_D60', 
    'B01_C10_D10', 
    'B01_C10_D20', 
    'B01_C10_D30', 
    'B01_C10_D40', 
    'B01_C10_D50', 
    'B01_C10_D60', 
    'B01_C11_D10', 
    'B01_C11_D20', 
    'B01_C11_D30', 
    'B01_C11_D40', 
    'B01_C11_D50', 
    'B01_C11_D60', 
    'B01_C12_D10', 
    'B01_C12_D20', 
    'B01_C12_D30', 
    'B01_C12_D40', 
    'B01_C12_D50', 
    'B01_C12_D60', 
    'B01_C13_D10', 
    'B01_C13_D20', 
    'B01_C13_D30', 
    'B01_C13_D40', 
    'B01_C13_D50', 
    'B01_C13_D60', 
    'B01_C14_D10', 
    'B01_C14_D20', 
    'B01_C14_D30', 
    'B01_C14_D40', 
    'B01_C14_D50', 
    'B01_C14_D60', 
    'B01_C15_D10', 
    'B01_C15_D20', 
    'B01_C15_D30', 
    'B01_C15_D40', 
    'B01_C15_D50', 
    'B01_C15_D60', 
    'B01_C16_D10', 
    'B01_C16_D20', 
    'B01_C16_D30', 
    'B01_C16_D40', 
    'B01_C16_D50', 
    'B01_C16_D60', 
    'B01_C17_D10', 
    'B01_C17_D20', 
    'B01_C17_D30', 
    'B01_C17_D40', 
    'B01_C17_D50', 
    'B01_C17_D60', 
    'B01_C18_D10', 
    'B01_C18_D20', 
    'B01_C18_D30', 
    'B01_C18_D40', 
    'B01_C18_D50', 
    'B01_C18_D60', 
    'B01_C19_D10', 
    'B01_C19_D20', 
    'B01_C19_D30', 
    'B01_C19_D40', 
    'B01_C19_D50', 
    'B01_C19_D60', 
    'B01_C20_D10', 
    'B01_C20_D20', 
    'B01_C20_D30', 
    'B01_C20_D40', 
    'B01_C20_D50', 
    'B01_C20_D60', 
    'B01_C21_D10', 
    'B01_C21_D20', 
    'B01_C21_D30', 
    'B01_C21_D40', 
    'B01_C21_D50', 
    'B01_C21_D60', 
    'B01_C22_D10', 
    'B01_C22_D20', 
    'B01_C22_D30', 
    'B01_C22_D40', 
    'B01_C22_D50', 
    'B01_C22_D60', 
    'B01_C23_D10', 
    'B01_C23_D20', 
    'B01_C23_D30', 
    'B01_C23_D40', 
    'B01_C23_D50', 
    'B01_C23_D60', 
    'B01_C24_D10', 
    'B01_C24_D20', 
    'B01_C24_D30', 
    'B01_C24_D40', 
    'B01_C24_D50', 
    'B01_C24_D60', 
    'B01_C25_D10', 
    'B01_C25_D20', 
    'B01_C25_D30', 
    'B01_C25_D40', 
    'B01_C25_D50', 
    'B01_C25_D60', 
    'B02_C01_D10', 
    'B02_C01_D20', 
    'B02_C01_D30', 
    'B02_C01_D40', 
    'B02_C01_D50', 
    'B02_C01_D60', 
    'B02_C02_D10', 
    'B02_C02_D20', 
    'B02_C02_D30', 
    'B02_C02_D40', 
    'B02_C02_D50', 
    'B02_C02_D60', 
    'B02_C03_D10', 
    'B02_C03_D20', 
    'B02_C03_D30', 
    'B02_C03_D40', 
    'B02_C03_D50', 
    'B02_C03_D60', 
    'B02_C04_D10', 
    'B02_C04_D20', 
    'B02_C04_D30', 
    'B02_C04_D40', 
    'B02_C04_D50', 
    'B02_C04_D60', 
    'B02_C05_D10', 
    'B02_C05_D20', 
    'B02_C05_D30', 
    'B02_C05_D40', 
    'B02_C05_D50', 
    'B02_C05_D60', 
    'B02_C06_D10', 
    'B02_C06_D20', 
    'B02_C06_D30', 
    'B02_C06_D40', 
    'B02_C06_D50', 
    'B02_C06_D60', 
    'B02_C07_D10', 
    'B02_C07_D20', 
    'B02_C07_D30', 
    'B02_C07_D40', 
    'B02_C07_D50', 
    'B02_C07_D60', 
    'B02_C08_D10', 
    'B02_C08_D20', 
    'B02_C08_D30', 
    'B02_C08_D40', 
    'B02_C08_D50', 
    'B02_C08_D60', 
    'B02_C09_D10', 
    'B02_C09_D20', 
    'B02_C09_D30', 
    'B02_C09_D40', 
    'B02_C09_D50', 
    'B02_C09_D60', 
    'B02_C10_D10', 
    'B02_C10_D20', 
    'B02_C10_D30', 
    'B02_C10_D40', 
    'B02_C10_D50', 
    'B02_C10_D60', 
    'B02_C11_D10', 
    'B02_C11_D20', 
    'B02_C11_D30', 
    'B02_C11_D40', 
    'B02_C11_D50', 
    'B02_C11_D60', 
    'B02_C12_D10', 
    'B02_C12_D20', 
    'B02_C12_D30', 
    'B02_C12_D40', 
    'B02_C12_D50', 
    'B02_C12_D60', 
    'B02_C13_D10', 
    'B02_C13_D20', 
    'B02_C13_D30', 
    'B02_C13_D40', 
    'B02_C13_D50', 
    'B02_C13_D60', 
    'B03_C01_D10', 
    'B03_C01_D20', 
    'B03_C01_D30', 
    'B03_C01_D40', 
    'B03_C01_D50', 
    'B03_C01_D60', 
    'B03_C02_D10', 
    'B03_C02_D20', 
    'B03_C02_D30', 
    'B03_C02_D40', 
    'B03_C02_D50', 
    'B03_C02_D60', 
    'B03_C03_D10', 
    'B03_C03_D20', 
    'B03_C03_D30', 
    'B03_C03_D40', 
    'B03_C03_D50', 
    'B03_C03_D60', 
    'B03_C04_D10', 
    'B03_C04_D20', 
    'B03_C04_D30', 
    'B03_C04_D40', 
    'B03_C04_D50', 
    'B03_C04_D60', 
    'B03_C05_D10', 
    'B03_C05_D20', 
    'B03_C05_D30', 
    'B03_C05_D40', 
    'B03_C05_D50', 
    'B03_C05_D60', 
    'B03_C06_D10', 
    'B03_C06_D20', 
    'B03_C06_D30', 
    'B03_C06_D40', 
    'B03_C06_D50', 
    'B03_C06_D60', 
    'B03_C07_D10', 
    'B03_C07_D20', 
    'B03_C07_D30', 
    'B03_C07_D40', 
    'B03_C07_D50', 
    'B03_C07_D60', 
    'B03_C08_D10', 
    'B03_C08_D20', 
    'B03_C08_D30', 
    'B03_C08_D40', 
    'B03_C08_D50', 
    'B03_C08_D60', 
    'B03_C09_D10', 
    'B03_C09_D20', 
    'B03_C09_D30', 
    'B03_C09_D40', 
    'B03_C09_D50', 
    'B03_C09_D60', 
    'B04_C01_D10', 
    'B04_C01_D20', 
    'B04_C01_D30', 
    'B04_C01_D40', 
    'B04_C01_D50', 
    'B04_C01_D60', 
    'B04_C02_D10', 
    'B04_C02_D20', 
    'B04_C02_D30', 
    'B04_C02_D40', 
    'B04_C02_D50', 
    'B04_C02_D60', 
    'B04_C03_D10', 
    'B04_C03_D20', 
    'B04_C03_D30', 
    'B04_C03_D40', 
    'B04_C03_D50', 
    'B04_C03_D60', 
    'B04_C04_D10', 
    'B04_C04_D20', 
    'B04_C04_D30', 
    'B04_C04_D40', 
    'B04_C04_D50', 
    'B04_C04_D60', 
    'B04_C05_D10', 
    'B04_C05_D20', 
    'B04_C05_D30', 
    'B04_C05_D40', 
    'B04_C05_D50', 
    'B04_C05_D60', 
    'B04_C06_D10', 
    'B04_C06_D20', 
    'B04_C06_D30', 
    'B04_C06_D40', 
    'B04_C06_D50', 
    'B04_C06_D60', 
    'B04_C07_D10', 
    'B04_C07_D20', 
    'B04_C07_D30', 
    'B04_C07_D40', 
    'B04_C07_D50', 
    'B04_C07_D60', 
    'B05_C01_D10', 
    'B05_C01_D20', 
    'B05_C01_D30', 
    'B05_C01_D40', 
    'B05_C01_D50', 
    'B05_C01_D60', 
    'B05_C02_D10', 
    'B05_C02_D20', 
    'B05_C02_D30', 
    'B05_C02_D40', 
    'B05_C02_D50', 
    'B05_C02_D60', 
    'B05_C03_D10', 
    'B05_C03_D20', 
    'B05_C03_D30', 
    'B05_C03_D40', 
    'B05_C03_D50', 
    'B05_C03_D60', 
    'B05_C04_D10', 
    'B05_C04_D20', 
    'B05_C04_D30', 
    'B05_C04_D40', 
    'B05_C04_D50', 
    'B05_C04_D60', 
    'B05_C05_D10', 
    'B05_C05_D20', 
    'B05_C05_D30', 
    'B05_C05_D40', 
    'B05_C05_D50', 
    'B05_C05_D60', 
    'B06_C01_D10', 
    'B06_C01_D20', 
    'B06_C01_D30', 
    'B06_C01_D40', 
    'B06_C01_D50', 
    'B06_C01_D60', 
    'B06_C02_D10', 
    'B06_C02_D20', 
    'B06_C02_D30', 
    'B06_C02_D40', 
    'B06_C02_D50', 
    'B06_C02_D60', 
    'B06_C03_D10', 
    'B06_C03_D20', 
    'B06_C03_D30', 
    'B06_C03_D40', 
    'B06_C03_D50', 
    'B06_C03_D60', 
    'B06_C04_D10', 
    'B06_C04_D20', 
    'B06_C04_D30', 
    'B06_C04_D40', 
    'B06_C04_D50', 
    'B06_C04_D60', 
    'B06_C05_D10', 
    'B06_C05_D20', 
    'B06_C05_D30', 
    'B06_C05_D40', 
    'B06_C05_D50', 
    'B06_C05_D60', 
    'B06_C06_D10', 
    'B06_C06_D20', 
    'B06_C06_D30', 
    'B06_C06_D40', 
    'B06_C06_D50', 
    'B06_C06_D60', 
    'B06_C07_D10', 
    'B06_C07_D20', 
    'B06_C07_D30', 
    'B06_C07_D40', 
    'B06_C07_D50', 
    'B06_C07_D60', 
    'B06_C08_D10', 
    'B06_C08_D20', 
    'B06_C08_D30', 
    'B06_C08_D40', 
    'B06_C08_D50', 
    'B06_C08_D60', 
    'B06_C09_D10', 
    'B06_C09_D20', 
    'B06_C09_D30', 
    'B06_C09_D40', 
    'B06_C09_D50', 
    'B06_C09_D60', 
    'B06_C10_D10', 
    'B06_C10_D20', 
    'B06_C10_D30', 
    'B06_C10_D40', 
    'B06_C10_D50', 
    'B06_C10_D60', 
    'B07_C01_D10', 
    'B07_C01_D20', 
    'B07_C01_D30', 
    'B07_C01_D40', 
    'B07_C01_D50', 
    'B07_C01_D60', 
    'B07_C02_D10', 
    'B07_C02_D20', 
    'B07_C02_D30', 
    'B07_C02_D40', 
    'B07_C02_D50', 
    'B07_C02_D60', 
    'B07_C03_D10', 
    'B07_C03_D20', 
    'B07_C03_D30', 
    'B07_C03_D40', 
    'B07_C03_D50', 
    'B07_C03_D60', 
    'B07_C04_D10', 
    'B07_C04_D20', 
    'B07_C04_D30', 
    'B07_C04_D40', 
    'B07_C04_D50', 
    'B07_C04_D60', 
    'B07_C05_D10', 
    'B07_C05_D20', 
    'B07_C05_D30', 
    'B07_C05_D40', 
    'B07_C05_D50', 
    'B07_C05_D60', 
    'B07_C06_D10', 
    'B07_C06_D20', 
    'B07_C06_D30', 
    'B07_C06_D40', 
    'B07_C06_D50', 
    'B07_C06_D60', 
    'B07_C07_D10', 
    'B07_C07_D20', 
    'B07_C07_D30', 
    'B07_C07_D40', 
    'B07_C07_D50', 
    'B07_C07_D60', 
    'B07_C08_D10', 
    'B07_C08_D20', 
    'B07_C08_D30', 
    'B07_C08_D40', 
    'B07_C08_D50', 
    'B07_C08_D60', 
    'B07_C09_D10', 
    'B07_C09_D20', 
    'B07_C09_D30', 
    'B07_C09_D40', 
    'B07_C09_D50', 
    'B07_C09_D60', 
    'B08_C01_D10', 
    'B08_C01_D20', 
    'B08_C01_D30', 
    'B08_C01_D40', 
    'B08_C01_D50', 
    'B08_C01_D60', 
    'B08_C02_D10', 
    'B08_C02_D20', 
    'B08_C02_D30', 
    'B08_C02_D40', 
    'B08_C02_D50', 
    'B08_C02_D60', 
    'B08_C03_D10', 
    'B08_C03_D20', 
    'B08_C03_D30', 
    'B08_C03_D40', 
    'B08_C03_D50', 
    'B08_C03_D60', 
    'B08_C04_D10', 
    'B08_C04_D20', 
    'B08_C04_D30', 
    'B08_C04_D40', 
    'B08_C04_D50', 
    'B08_C04_D60', 
    'B08_C05_D10', 
    'B08_C05_D20', 
    'B08_C05_D30', 
    'B08_C05_D40', 
    'B08_C05_D50', 
    'B08_C05_D60', 
    'B08_C06_D10', 
    'B08_C06_D20', 
    'B08_C06_D30', 
    'B08_C06_D40', 
    'B08_C06_D50', 
    'B08_C06_D60', 
    'B08_C07_D10', 
    'B08_C07_D20', 
    'B08_C07_D30', 
    'B08_C07_D40', 
    'B08_C07_D50', 
    'B08_C07_D60', 
    'B09_C01_D10', 
    'B09_C01_D20', 
    'B09_C01_D30', 
    'B09_C01_D40', 
    'B09_C01_D50', 
    'B09_C01_D60', 
    'B09_C02_D10', 
    'B09_C02_D20', 
    'B09_C02_D30', 
    'B09_C02_D40', 
    'B09_C02_D50', 
    'B09_C02_D60', 
    'B09_C03_D10', 
    'B09_C03_D20', 
    'B09_C03_D30', 
    'B09_C03_D40', 
    'B09_C03_D50', 
    'B09_C03_D60', 
    'B09_C04_D10', 
    'B09_C04_D20', 
    'B09_C04_D30', 
    'B09_C04_D40', 
    'B09_C04_D50', 
    'B09_C04_D60', 
    'B09_C05_D10', 
    'B09_C05_D20', 
    'B09_C05_D30', 
    'B09_C05_D40', 
    'B09_C05_D50', 
    'B09_C05_D60', 
    'B09_C06_D10', 
    'B09_C06_D20', 
    'B09_C06_D30', 
    'B09_C06_D40', 
    'B09_C06_D50', 
    'B09_C06_D60', 
    'B09_C07_D10', 
    'B09_C07_D20', 
    'B09_C07_D30', 
    'B09_C07_D40', 
    'B09_C07_D50', 
    'B09_C07_D60', 
    'B09_C08_D10', 
    'B09_C08_D20', 
    'B09_C08_D30', 
    'B09_C08_D40', 
    'B09_C08_D50', 
    'B09_C08_D60', 
    'B09_C09_D10', 
    'B09_C09_D20', 
    'B09_C09_D30', 
    'B09_C09_D40', 
    'B09_C09_D50', 
    'B09_C09_D60', 
    'B10_C01_D10', 
    'B10_C01_D20', 
    'B10_C01_D30', 
    'B10_C01_D40', 
    'B10_C01_D50', 
    'B10_C01_D60', 
    'B10_C02_D10', 
    'B10_C02_D20', 
    'B10_C02_D30', 
    'B10_C02_D40', 
    'B10_C02_D50', 
    'B10_C02_D60', 
    'B10_C03_D10', 
    'B10_C03_D20', 
    'B10_C03_D30', 
    'B10_C03_D40', 
    'B10_C03_D50', 
    'B10_C03_D60', 
    'B10_C04_D10', 
    'B10_C04_D20', 
    'B10_C04_D30', 
    'B10_C04_D40', 
    'B10_C04_D50', 
    'B10_C04_D60', 
    'B10_C05_D10', 
    'B10_C05_D20', 
    'B10_C05_D30', 
    'B10_C05_D40', 
    'B10_C05_D50', 
    'B10_C05_D60', 
    'B10_C06_D10', 
    'B10_C06_D20', 
    'B10_C06_D30', 
    'B10_C06_D40', 
    'B10_C06_D50', 
    'B10_C06_D60'
)