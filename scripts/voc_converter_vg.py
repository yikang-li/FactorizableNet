#!/usr/bin/env python
'''
This file is a tool to parse json file and generate voc format xml file.
'''
import json
import xml.etree.ElementTree as ET
import cv2
import os
import os.path as osp
import pdb


def main():
  base_data_dir = '/DATA/ykli/workspace/scene_generation/data/visual_genome/vg_cleansing/output/top_150_50_new'
  out_xml_path = osp.join(base_data_dir, "object_xml")

  if not osp.isdir(out_xml_path):
      os.makedirs(out_xml_path)

  annotations = json.load(open(osp.join(base_data_dir, "test.json")))

  #jpg files folder
  counter = 0

  for i in range(len(annotations)):
      jpg_name = annotations[i]['path']
      xml_file_name = os.path.splitext(jpg_name)[0] + ".xml"
      im_height = annotations[i]['height']
      im_width = annotations[i]['width']
      im_ch = 3
      counter += 1

      #create a xml
      out = ET.Element('annotation')
      #folder
      folder = ET.SubElement(out,"folder")
      folder.text = "VOC2007"
      #filename
      filename = ET.SubElement(out,"filename")
      filename.text = jpg_name
      #filesource
      file_source = ET.SubElement(out,"source")
      database = ET.SubElement(file_source,"database")
      database.text = "VRD Database"
      annotation = ET.SubElement(file_source,"annotation")
      annotation.text = "VRD"
      image = ET.SubElement(file_source,"image")
      image.text = "flickr"
      flickid = ET.SubElement(file_source,"flickrid")
      flickid.text = "Yikang"

      #file owner
      owner = ET.SubElement(out,"owner")
      flickid = ET.SubElement(owner,"flickrid")
      flickid.text = "Yikang"
      name = ET.SubElement(owner,"name")
      name.text = "Yikang"

      #file size
      file_size = ET.SubElement(out,"size")
      file_width = ET.SubElement(file_size,"width")
      file_width.text = str(im_height)
      file_height = ET.SubElement(file_size,"height")
      file_height.text = str(im_width)
      file_depth = ET.SubElement(file_size,"depth")
      file_depth.text = str(im_ch)

      #file segmented
      file_segmented = ET.SubElement(out,"segmented")
      file_segmented.text = "0"

      for obj in annotations[i]['objects']:
          bbox_x1 = obj['box'][0]
          bbox_y1 = obj['box'][1]
          bbox_x2 = obj['box'][2]
          bbox_y2 = obj['box'][3]
          obj_class = obj['class']
          #create a car obj
          obj = ET.SubElement(out,'object')
          obj_name = ET.SubElement(obj,"name")
          obj_name.text = obj_class

          obj_pose = ET.SubElement(obj,"pose")
          obj_pose.text = "Unspecified"

          obj_truncated = ET.SubElement(obj,"truncated")
          obj_truncated.text = "1"

          obj_difficult = ET.SubElement(obj,"difficult")
          obj_difficult.text = "0"

          #create boundingbox
          bndbox = ET.SubElement(obj,"bndbox")
          xmin = ET.SubElement(bndbox,'xmin')
          xmin.text = str(bbox_x1)

          ymin = ET.SubElement(bndbox,'ymin')
          ymin.text = str(bbox_y1)

          xmax = ET.SubElement(bndbox,'xmax')
          xmax.text = str(bbox_x2)

          ymax = ET.SubElement(bndbox,'ymax')
          ymax.text = str(bbox_y2)

      out_tree = ET.ElementTree(out)
      out_tree.write(osp.join(out_xml_path, xml_file_name))

      if (i+1) % 100 == 0:
          print('{} / {} images processed'.format(i+1, len(annotations)))

  print "Process done"


if __name__ == '__main__':
  main()
