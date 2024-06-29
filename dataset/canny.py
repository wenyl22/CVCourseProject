import cv2
import os

def get_canny_edges(dataroot, savepath):
    for root, _, files in os.walk(dataroot):
        for filename in files:
            print(filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                edges = cv2.Canny(img, 80, 150)
                save_img_path = os.path.join(savepath, filename)
                cv2.imwrite(save_img_path, edges)

def main():
    dataroot = "./dataset/photo2paint/test_B"
    savepath = "./dataset/photo2paint/test_B_canny"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    get_canny_edges(dataroot, savepath)
        
if __name__ == "__main__":
    main()