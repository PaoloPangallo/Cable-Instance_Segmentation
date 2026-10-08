import math
import unittest
import numpy as np
from cable_seg.core import (ProjectConfig, inspect_coco, line_from_mask,
                            axial_angle_error_deg, mask_iou, match_masks,
                            select_by_score)

class CoreTests(unittest.TestCase):
    def test_coco(self):
        data={"images":[{"id":1},{"id":2}],
              "annotations":[{"image_id":1}],"categories":[{"id":1}]}
        self.assertEqual(inspect_coco(data)["images_without_annotations"],1)
        self.assertEqual(inspect_coco(data)["mean_instances_per_image"],0.5)

    def test_lines(self):
        h=np.zeros((60,60)); h[10,5:55]=1
        v=np.zeros((60,60)); v[5:55,20]=1
        a=line_from_mask(h); b=line_from_mask(v)
        self.assertAlmostEqual(axial_angle_error_deg(a[1],b[1]),90)
        self.assertAlmostEqual(axial_angle_error_deg(a[1],a[1]+math.pi),0)

    def test_short(self):
        self.assertIsNone(line_from_mask(np.ones((3,3))))

    def test_one_to_one(self):
        m=np.zeros((6,6));m[1:4,1:4]=1
        self.assertEqual(mask_iou(m,m),1.0)
        self.assertEqual(len(match_masks([m,m],[m])),1)

    def test_threshold(self):
        self.assertEqual(len(select_by_score([{"score":0.2},{"score":0.5}],0.5)),1)
        with self.assertRaises(ValueError): select_by_score([],1.2)
        with self.assertRaises(ValueError): ProjectConfig().ann_path("validation")

if __name__=="__main__":
    unittest.main()
