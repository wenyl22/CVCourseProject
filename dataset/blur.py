import cv2
import os

def apply_gaussian_blur_on_edges(dataroot, savepath):
    for root, _, files in os.walk(dataroot):
        for filename in files:
            print(filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                
                edges = cv2.Canny(img, 100, 200)
                mask = edges != 0
                dst = img.copy()
                
                blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
                dst[mask] = blurred_img[mask]
                
                save_img_path = os.path.join(savepath, filename)
                cv2.imwrite(save_img_path, dst)


def main():
    dataroot = "D:\\course\\G2Spring\\Computer Vision\\CourseProject\\Chinese-Landscape-Painting-Dataset\\All-Paintings\\train_A"
    savepath = "D:\\course\\G2Spring\\Computer Vision\\CourseProject\\Chinese-Landscape-Painting-Dataset\\All-Paintings\\train_A_blurred"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    apply_gaussian_blur_on_edges(dataroot, savepath)
        
if __name__ == "__main__":
    main()