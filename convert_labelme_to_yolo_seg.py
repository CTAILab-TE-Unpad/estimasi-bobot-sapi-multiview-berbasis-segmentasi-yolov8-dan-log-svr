import os
import json
import glob
import math
import numpy as np
import cv2

def circle_to_polygon(center, point_on_perimeter, num_points=32):
    """Mengubah bentuk circle Labelme (center & radius point) menjadi N-titik polygon."""
    cx, cy = center
    px, py = point_on_perimeter
    radius = math.hypot(px - cx, py - cy)
    
    polygon = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        polygon.append([x, y])
    return polygon

def convert_labelme_json_to_yolo_seg(json_dir, output_dir, class_mapping={"sticker": 0}, num_circle_points=32):
    """
    Konversi seluruh file JSON dari Labelme (baik tipe polygon maupun circle)
    ke format TXT YOLOv8 Instance Segmentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Ditemukan {len(json_files)} file JSON Labelme di '{json_dir}'.")
    converted_count = 0
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        
        txt_filename = os.path.splitext(os.path.basename(jf))[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            shape_type = shape.get("shape_type", "polygon")
            points = shape.get("points", [])
            
            if label not in class_mapping:
                continue
            class_id = class_mapping[label]
            
            if shape_type == "circle" and len(points) == 2:
                poly_pts = circle_to_polygon(points[0], points[1], num_points=num_circle_points)
            elif shape_type == "polygon":
                poly_pts = points
            else:
                continue
                
            # Normalisasi koordinat ke [0, 1]
            norm_coords = []
            for pt in poly_pts:
                nx = max(0.0, min(1.0, pt[0] / img_w))
                ny = max(0.0, min(1.0, pt[1] / img_h))
                norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                
            line = f"{class_id} " + " ".join(norm_coords)
            lines.append(line)
            
        with open(txt_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(lines) + "\n")
            
        converted_count += 1
        
    print(f"Berhasil mengonversi {converted_count} file ke format YOLOv8 segmentation di '{output_dir}'.")

if __name__ == "__main__":
    # Contoh pemakaian:
    import argparse
    parser = argparse.ArgumentParser(description="Convert Labelme JSON to YOLOv8 Segmentation TXT")
    parser.add_argument("--json_dir", type=str, default="datasets/sticker_circle/images", help="Folder berisi file .json Labelme")
    parser.add_argument("--output_dir", type=str, default="datasets/sticker_circle/labels", help="Folder output file .txt YOLO")
    parser.add_argument("--class_name", type=str, default="sticker", help="Nama label class di Labelme")
    args = parser.parse_args()
    
    convert_labelme_json_to_yolo_seg(args.json_dir, args.output_dir, class_mapping={args.class_name: 0})
