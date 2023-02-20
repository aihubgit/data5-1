# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import csv

from . import general

class_index = {352: 0, 353: 1, 354: 2, 355: 3, 356: 4, 357: 5, 358: 6, 359: 7, 360: 8, 361: 9, 362: 10, 363: 11, 364: 12, 365: 13, 366: 14, 367: 15, 368: 16, 369: 17, 370: 18, 371: 19, 372: 20, 373: 21, 374: 22, 375: 23, 376: 24, 377: 25, 378: 26, 379: 27, 380: 28, 381: 29, 382: 30, 383: 31, 384: 32, 385: 33, 386: 34, 387: 35, 388: 36, 389: 37, 390: 38, 391: 39, 392: 40, 393: 41, 394: 42, 395: 43, 396: 44, 397: 45, 398: 46, 399: 47, 400: 48, 401: 49, 402: 50, 403: 51, 404: 52, 405: 53, 406: 54, 407: 55, 408: 56, 409: 57, 410: 58, 411: 59, 412: 60, 413: 61, 414: 62, 415: 63, 416: 64, 417: 65, 418: 66, 419: 67, 420: 68, 421: 69, 422: 70, 423: 71, 424: 72, 425: 73, 426: 74, 427: 75, 428: 76, 429: 77, 430: 78, 431: 79, 432: 80, 433: 81, 434: 82, 435: 83, 436: 84, 437: 85, 438: 86, 439: 87, 440: 88, 441: 89, 442: 90, 443: 91, 444: 92, 445: 93, 446: 94, 447: 95, 448: 96, 449: 97, 450: 98, 451: 99, 452: 100, 453: 101, 454: 102, 455: 103, 456: 104, 457: 105, 458: 106, 459: 107, 460: 108, 461: 109, 462: 110, 463: 111, 464: 112, 465: 113, 466: 114, 467: 115, 468: 116, 469: 117, 470: 118, 471: 119, 472: 120, 473: 121, 474: 122, 475: 123, 476: 124, 477: 125, 478: 126, 479: 127, 480: 128, 481: 129, 482: 130, 483: 131, 484: 132, 485: 133, 486: 134, 487: 135, 488: 136, 489: 137, 490: 138, 491: 139, 492: 140, 493: 141, 494: 142, 495: 143, 496: 144, 497: 145, 498: 146, 499: 147, 500: 148, 501: 149, 502: 150, 503: 151, 504: 152, 505: 153, 506: 154, 507: 155, 508: 156, 509: 157, 510: 158, 511: 159, 512: 160, 513: 161, 514: 162, 515: 163, 516: 164, 517: 165, 518: 166, 519: 167, 520: 168, 521: 169, 522: 170, 523: 171, 524: 172, 525: 173, 526: 174, 527: 175, 528: 176, 529: 177, 530: 178, 531: 179, 532: 180, 533: 181, 534: 182, 535: 183, 536: 184, 537: 185, 538: 186, 539: 187, 540: 188, 541: 189, 542: 190, 543: 191, 544: 192, 545: 193, 546: 194, 547: 195, 548: 196, 549: 197, 550: 198, 551: 199, 552: 200, 553: 201, 554: 202, 555: 203, 556: 204, 557: 205, 558: 206, 559: 207, 560: 208, 561: 209, 562: 210, 563: 211, 564: 212, 565: 213, 566: 214, 567: 215, 568: 216, 569: 217, 570: 218, 571: 219, 572: 220, 573: 221, 574: 222, 575: 223, 576: 224, 577: 225, 578: 226, 579: 227, 580: 228, 581: 229, 582: 230, 583: 231, 584: 232, 585: 233, 586: 234, 587: 235, 588: 236, 589: 237, 590: 238, 591: 239, 592: 240, 593: 241, 594: 242, 595: 243, 596: 244, 597: 245, 598: 246, 599: 247, 600: 248, 601: 249, 602: 250, 603: 251, 604: 252, 605: 253, 606: 254, 607: 255, 608: 256, 609: 257, 610: 258, 611: 259, 612: 260, 613: 261, 614: 262, 615: 263, 616: 264, 617: 265, 618: 266, 619: 267, 620: 268, 621: 269, 622: 270, 623: 271, 624: 272, 625: 273, 626: 274, 627: 275, 628: 276, 629: 277, 630: 278, 631: 279, 632: 280, 633: 281, 634: 282, 635: 283, 636: 284, 637: 285, 638: 286, 639: 287, 640: 288, 641: 289, 642: 290, 643: 291, 644: 292, 645: 293, 646: 294, 647: 295, 648: 296, 649: 297, 650: 298, 651: 299, 652: 300, 653: 301, 654: 302, 655: 303, 656: 304, 657: 305, 658: 306, 659: 307, 660: 308, 661: 309, 662: 310, 663: 311, 664: 312, 665: 313, 666: 314, 667: 315, 668: 316, 669: 317, 670: 318, 671: 319, 672: 320, 673: 321, 674: 322, 675: 323, 676: 324, 677: 325, 678: 326, 679: 327, 680: 328, 681: 329, 682: 330, 683: 331, 684: 332, 685: 333, 686: 334, 687: 335, 688: 336, 689: 337, 690: 338, 691: 339, 692: 340, 693: 341, 694: 342, 695: 343, 696: 344, 697: 345, 698: 346, 699: 347, 700: 348, 701: 349, 702: 350, 703: 351, 704: 352, 705: 353, 706: 354, 707: 355, 708: 356, 709: 357, 710: 358, 711: 359, 712: 360, 713: 361, 714: 362, 715: 363, 716: 364, 717: 365, 718: 366, 719: 367, 720: 368, 721: 369, 722: 370, 723: 371, 724: 372, 725: 373, 726: 374, 727: 375, 728: 376, 729: 377, 730: 378, 731: 379, 732: 380, 733: 381, 734: 382, 735: 383, 736: 384, 737: 385, 738: 386, 739: 387, 740: 388, 741: 389, 742: 390, 743: 391, 744: 392, 745: 393, 746: 394, 747: 395, 748: 396, 749: 397, 750: 398, 751: 399, 752: 400, 753: 401, 754: 402, 755: 403, 756: 404, 757: 405, 758: 406, 759: 407, 760: 408, 761: 409, 762: 410, 763: 411, 764: 412, 765: 413, 766: 414, 767: 415, 768: 416, 769: 417, 770: 418, 771: 419, 772: 420, 773: 421, 774: 422, 775: 423, 776: 424, 777: 425, 778: 426, 779: 427, 780: 428, 781: 429, 782: 430, 783: 431, 784: 432, 785: 433, 786: 434, 787: 435, 788: 436, 789: 437, 790: 438, 791: 439, 792: 440, 793: 441, 794: 442, 795: 443, 796: 444, 797: 445, 798: 446, 799: 447, 800: 448, 801: 449, 802: 450, 803: 451, 804: 452, 805: 453, 806: 454, 807: 455, 808: 456, 809: 457, 810: 458, 811: 459, 812: 460, 813: 461, 814: 462, 815: 463, 816: 464, 817: 465, 818: 466, 819: 467, 820: 468, 821: 469, 822: 470, 823: 471, 824: 472, 825: 473, 826: 474, 827: 475, 828: 476, 829: 477, 830: 478, 831: 479, 832: 480, 833: 481, 834: 482, 835: 483, 836: 484, 837: 485, 838: 486, 839: 487, 840: 488, 841: 489, 842: 490, 843: 491, 844: 492, 845: 493, 846: 494, 847: 495, 848: 496, 849: 497, 850: 498, 851: 499, 852: 500, 853: 501, 854: 502, 855: 503, 856: 504, 857: 505, 858: 506, 859: 507, 860: 508, 861: 509, 862: 510, 863: 511, 864: 512, 865: 513, 866: 514, 867: 515, 868: 516, 869: 517, 870: 518, 871: 519, 872: 520, 873: 521, 874: 522, 875: 523, 876: 524, 877: 525, 878: 526, 879: 527, 880: 528, 881: 529, 882: 530, 883: 531, 884: 532, 885: 533, 886: 534, 887: 535, 888: 536, 889: 537, 890: 538, 891: 539, 892: 540, 893: 541, 894: 542, 895: 543, 896: 544, 897: 545, 898: 546, 899: 547, 900: 548, 901: 549, 902: 550, 903: 551, 904: 552, 905: 553, 906: 554, 907: 555, 908: 556, 909: 557, 910: 558, 911: 559, 912: 560, 913: 561, 914: 562, 915: 563, 916: 564, 917: 565, 918: 566, 919: 567, 920: 568, 921: 569, 922: 570, 923: 571, 924: 572, 925: 573, 926: 574, 927: 575, 928: 576, 929: 577, 930: 578, 931: 579, 932: 580, 933: 581, 934: 582, 935: 583, 936: 584, 937: 585, 938: 586, 939: 587, 940: 588, 941: 589, 942: 590, 943: 591, 944: 592, 945: 593, 946: 594, 947: 595, 948: 596, 949: 597, 950: 598, 951: 599, 952 : 600}

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.csv = [['img_id', 'GT class', 'prediction class', 'confidence', 'Is correct', 'accumulated TRUE', 'accumulated FALSE']]

    def process_batch(self, detections, labels, img_index):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0]
        detection_classes = detections[:, 5]
        iou = general.box_iou(labels[:, 1:], detections[:, :4])
        print(gt_classes)
        # gt_classes === [[123], [313], [4124], [4124] ,[124]]{352: 0, 353: 1, 354: 2, 355: 3, 356: 4, 357: 5, 358: 6, 359: 7, 360: 8, 361: 9, 362: 10, 363: 11, 364: 12, 365: 13, 366: 14, 367: 15, 368: 16, 369: 17, 370: 18, 371: 19, 372: 20, 373: 21, 374: 22, 375: 23, 376: 24, 377: 25, 378: 26, 379: 27, 380: 28, 381: 29, 382: 30, 383: 31, 384: 32, 385: 33, 386: 34, 387: 35, 388: 36, 389: 37, 390: 38, 391: 39, 392: 40, 393: 41, 394: 42, 395: 43, 396: 44, 397: 45, 398: 46, 399: 47, 400: 48, 401: 49, 402: 50, 403: 51, 404: 52, 405: 53, 406: 54, 407: 55, 408: 56, 409: 57, 410: 58, 411: 59, 412: 60, 413: 61, 414: 62, 415: 63, 416: 64, 417: 65, 418: 66, 419: 67, 420: 68, 421: 69, 422: 70, 423: 71, 424: 72, 425: 73, 426: 74, 427: 75, 428: 76, 429: 77, 430: 78, 431: 79, 432: 80, 433: 81, 434: 82, 435: 83, 436: 84, 437: 85, 438: 86, 439: 87, 440: 88, 441: 89, 442: 90, 443: 91, 444: 92, 445: 93, 446: 94, 447: 95, 448: 96, 449: 97, 450: 98, 451: 99, 452: 100, 453: 101, 454: 102, 455: 103, 456: 104, 457: 105, 458: 106, 459: 107, 460: 108, 461: 109, 462: 110, 463: 111, 464: 112, 465: 113, 466: 114, 467: 115, 468: 116, 469: 117, 470: 118, 471: 119, 472: 120, 473: 121, 474: 122, 475: 123, 476: 124, 477: 125, 478: 126, 479: 127, 480: 128, 481: 129, 482: 130, 483: 131, 484: 132, 485: 133, 486: 134, 487: 135, 488: 136, 489: 137, 490: 138, 491: 139, 492: 140, 493: 141, 494: 142, 495: 143, 496: 144, 497: 145, 498: 146, 499: 147, 500: 148, 501: 149, 502: 150, 503: 151, 504: 152, 505: 153, 506: 154, 507: 155, 508: 156, 509: 157, 510: 158, 511: 159, 512: 160, 513: 161, 514: 162, 515: 163, 516: 164, 517: 165, 518: 166, 519: 167, 520: 168, 521: 169, 522: 170, 523: 171, 524: 172, 525: 173, 526: 174, 527: 175, 528: 176, 529: 177, 530: 178, 531: 179, 532: 180, 533: 181, 534: 182, 535: 183, 536: 184, 537: 185, 538: 186, 539: 187, 540: 188, 541: 189, 542: 190, 543: 191, 544: 192, 545: 193, 546: 194, 547: 195, 548: 196, 549: 197, 550: 198, 551: 199, 552: 200, 553: 201, 554: 202, 555: 203, 556: 204, 557: 205, 558: 206, 559: 207, 560: 208, 561: 209, 562: 210, 563: 211, 564: 212, 565: 213, 566: 214, 567: 215, 568: 216, 569: 217, 570: 218, 571: 219, 572: 220, 573: 221, 574: 222, 575: 223, 576: 224, 577: 225, 578: 226, 579: 227, 580: 228, 581: 229, 582: 230, 583: 231, 584: 232, 585: 233, 586: 234, 587: 235, 588: 236, 589: 237, 590: 238, 591: 239, 592: 240, 593: 241, 594: 242, 595: 243, 596: 244, 597: 245, 598: 246, 599: 247, 600: 248, 601: 249, 602: 250, 603: 251, 604: 252, 605: 253, 606: 254, 607: 255, 608: 256, 609: 257, 610: 258, 611: 259, 612: 260, 613: 261, 614: 262, 615: 263, 616: 264, 617: 265, 618: 266, 619: 267, 620: 268, 621: 269, 622: 270, 623: 271, 624: 272, 625: 273, 626: 274, 627: 275, 628: 276, 629: 277, 630: 278, 631: 279, 632: 280, 633: 281, 634: 282, 635: 283, 636: 284, 637: 285, 638: 286, 639: 287, 640: 288, 641: 289, 642: 290, 643: 291, 644: 292, 645: 293, 646: 294, 647: 295, 648: 296, 649: 297, 650: 298, 651: 299, 652: 300, 653: 301, 654: 302, 655: 303, 656: 304, 657: 305, 658: 306, 659: 307, 660: 308, 661: 309, 662: 310, 663: 311, 664: 312, 665: 313, 666: 314, 667: 315, 668: 316, 669: 317, 670: 318, 671: 319, 672: 320, 673: 321, 674: 322, 675: 323, 676: 324, 677: 325, 678: 326, 679: 327, 680: 328, 681: 329, 682: 330, 683: 331, 684: 332, 685: 333, 686: 334, 687: 335, 688: 336, 689: 337, 690: 338, 691: 339, 692: 340, 693: 341, 694: 342, 695: 343, 696: 344, 697: 345, 698: 346, 699: 347, 700: 348, 701: 349, 702: 350, 703: 351, 704: 352, 705: 353, 706: 354, 707: 355, 708: 356, 709: 357, 710: 358, 711: 359, 712: 360, 713: 361, 714: 362, 715: 363, 716: 364, 717: 365, 718: 366, 719: 367, 720: 368, 721: 369, 722: 370, 723: 371, 724: 372, 725: 373, 726: 374, 727: 375, 728: 376, 729: 377, 730: 378, 731: 379, 732: 380, 733: 381, 734: 382, 735: 383, 736: 384, 737: 385, 738: 386, 739: 387, 740: 388, 741: 389, 742: 390, 743: 391, 744: 392, 745: 393, 746: 394, 747: 395, 748: 396, 749: 397, 750: 398, 751: 399, 752: 400, 753: 401, 754: 402, 755: 403, 756: 404, 757: 405, 758: 406, 759: 407, 760: 408, 761: 409, 762: 410, 763: 411, 764: 412, 765: 413, 766: 414, 767: 415, 768: 416, 769: 417, 770: 418, 771: 419, 772: 420, 773: 421, 774: 422, 775: 423, 776: 424, 777: 425, 778: 426, 779: 427, 780: 428, 781: 429, 782: 430, 783: 431, 784: 432, 785: 433, 786: 434, 787: 435, 788: 436, 789: 437, 790: 438, 791: 439, 792: 440, 793: 441, 794: 442, 795: 443, 796: 444, 797: 445, 798: 446, 799: 447, 800: 448, 801: 449, 802: 450, 803: 451, 804: 452, 805: 453, 806: 454, 807: 455, 808: 456, 809: 457, 810: 458, 811: 459, 812: 460, 813: 461, 814: 462, 815: 463, 816: 464, 817: 465, 818: 466, 819: 467, 820: 468, 821: 469, 822: 470, 823: 471, 824: 472, 825: 473, 826: 474, 827: 475, 828: 476, 829: 477, 830: 478, 831: 479, 832: 480, 833: 481, 834: 482, 835: 483, 836: 484, 837: 485, 838: 486, 839: 487, 840: 488, 841: 489, 842: 490, 843: 491, 844: 492, 845: 493, 846: 494, 847: 495, 848: 496, 849: 497, 850: 498, 851: 499, 852: 500, 853: 501, 854: 502, 855: 503, 856: 504, 857: 505, 858: 506, 859: 507, 860: 508, 861: 509, 862: 510, 863: 511, 864: 512, 865: 513, 866: 514, 867: 515, 868: 516, 869: 517, 870: 518, 871: 519, 872: 520, 873: 521, 874: 522, 875: 523, 876: 524, 877: 525, 878: 526, 879: 527, 880: 528, 881: 529, 882: 530, 883: 531, 884: 532, 885: 533, 886: 534, 887: 535, 888: 536, 889: 537, 890: 538, 891: 539, 892: 540, 893: 541, 894: 542, 895: 543, 896: 544, 897: 545, 898: 546, 899: 547, 900: 548, 901: 549, 902: 550, 903: 551, 904: 552, 905: 553, 906: 554, 907: 555, 908: 556, 909: 557, 910: 558, 911: 559, 912: 560, 913: 561, 914: 562, 915: 563, 916: 564, 917: 565, 918: 566, 919: 567, 920: 568, 921: 569, 922: 570, 923: 571, 924: 572, 925: 573, 926: 574, 927: 575, 928: 576, 929: 577, 930: 578, 931: 579, 932: 580, 933: 581, 934: 582, 935: 583, 936: 584, 937: 585, 938: 586, 939: 587, 940: 588, 941: 589, 942: 590, 943: 591, 944: 592, 945: 593, 946: 594, 947: 595, 948: 596, 949: 597, 950: 598, 951: 599}
        # asd = {}
        
        # gt_classes.sort()
        
        # for i, value in enumerate(gt_classes):
        #     asd[value] = i

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        idx = 0
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                print(gc)
                print(class_index[int(detection_classes[m1[j]])])
                print(self.matrix[gc, class_index[int(detection_classes[m1[j]])]])
                self.matrix[class_index[gc], class_index[int(detection_classes[m1[j]])]] += 1  # correct
                print(f"---------------------------------GC: {gc}")
                
                if gc.item() != detection_classes[m1[j]].item(): 
                        log_list = [img_index, gc.item(), int(detection_classes[m1[j]].item()), detections[int(matches.item((idx, 1))), 4].item(), 'False', '', self.matrix[class_index[int(detection_classes[m1[j]])], class_index[gc]].item()]
                        print(log_list)
                        self.csv.append(log_list)
                else:
                    print(img_index)
                    print(gc.item())
                    print(detection_classes[m1[j]].item())
                    print(detections[int(matches.item((idx, 1))), 4].item())
                    print('TRUE')
                    print(self.matrix[class_index[int(detection_classes[m1[j]])], class_index[gc]].item())
                    log_list = [img_index, gc.item(), int(detection_classes[m1[j]].item()), detections[int(matches.item((idx, 1))), 4].item(), 'TRUE', self.matrix[class_index[int(detection_classes[m1[j]])], class_index[gc]].item(), '']
                    print(log_list)
                    self.csv.append(log_list)
                    print("--->>> img_id: {} --GroundTruth: {} --Predict: {} --IoU: {:.3f} --Confidence: {:.3f} --Is correct?: {} --accumulated TRUE: {} -- accumulated FALSE: {}"
                    .format(log_list[0], log_list[1], log_list[2], matches[idx, 2], log_list[3], log_list[4], log_list[5], log_list[6]))

                    idx += 1

            else:
                self.matrix[class_index[self.nc], class_index[gc]] += 1  # background FP
                log_list = [img_index, gc.item(), self.nc, '', 'FP', '', self.matrix[class_index[self.nc], class_index[gc]].item()]
                self.csv.append(log_list)

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    print(int(dc))
                    print(self.nc)
                    self.matrix[class_index[int(dc)], class_index[self.nc]] += 1  # background FN
                    print(f"---------------------------------DC: {int(dc)}")
                    log_list = [img_index, self.nc, dc.item(), '', 'FN', '', self.matrix[class_index[int(dc)], class_index[self.nc]].item()]
                    self.csv.append(log_list)

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = general.box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    # px, py = np.linspace(0, 1, 1000), []  # for plotting
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            # recall = tpc / (n_l + 1e-16)  # recall curve
            recall = tpc / (n_l)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    # f1 = 2 * p * r / (p + r + 1e-16)
    f1 = 2 * p * r / (p + r)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names=[])
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names=[], ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names=[], ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names=[], ylabel='Recall')
        # plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names=["Signage_L0","Window_L0","Tree_L0", "Facilities_L0", "Greenhouse_L0", "Signage_L2","Window_L2","Tree_L2", "Facilities_L2", "Greenhouse_L2", "TP_Partially_Broken_L2", "TP_Fully_Broken_L2", "TP_Flood_Damage_L2"])
        # plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names=["Signage_L0","Window_L0","Tree_L0", "Facilities_L0", "Greenhouse_L0", "Signage_L2","Window_L2","Tree_L2", "Facilities_L2", "Greenhouse_L2", "TP_Partially_Broken_L2", "TP_Fully_Broken_L2", "TP_Flood_Damage_L2"], ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names=["Signage_L0","Window_L0","Tree_L0", "Facilities_L0", "Greenhouse_L0", "Signage_L2","Window_L2","Tree_L2", "Facilities_L2", "Greenhouse_L2", "TP_Partially_Broken_L2", "TP_Fully_Broken_L2", "TP_Flood_Damage_L2"], ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names=["Signage_L0","Window_L0","Tree_L0", "Facilities_L0", "Greenhouse_L0", "Signage_L2","Window_L2","Tree_L2", "Facilities_L2", "Greenhouse_L2", "TP_Partially_Broken_L2", "TP_Fully_Broken_L2", "TP_Flood_Damage_L2"], ylabel='Recall')
        # plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names=["Building_L0","Piloti_L0","Fence_L0","Building_L2","Piloti_L2","Fence_L2","Crack-Damage", "Broken-Damage"])
        # plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names=["Building_L0","Piloti_L0","Fence_L0","Building_L2","Piloti_L2","Fence_L2","Crack-Damage", "Broken-Damage"], ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names=["Building_L0","Piloti_L0","Fence_L0","Building_L2","Piloti_L2","Fence_L2","Crack-Damage", "Broken-Damage"], ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names=["Building_L0","Piloti_L0","Fence_L0","Building_L2","Piloti_L2","Fence_L2","Crack-Damage", "Broken-Damage"], ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')