import cv2


img = cv2.imread(r"C:\Users\Naman\Desktop\CSE676-FinalProject\image_processing\davis.jpg")
#print(img)
#cv2.imshow('Kakshi', img)
#cv2.waitKey(2000)

print(img.shape)

resized_img=cv2.resize(img, (600, 600))
#cv2.imshow('Kakshi', resized_img)
#cv2.waitKey(2000)

rgb_2_bw=cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img',rgb_2_bw)
#cv2.waitKey(2000)

#invert_edge=255-rgb_2_bw
#cv2.imshow('img',invert_edge)
#cv2.waitKey(2000)

img_invert = cv2.bitwise_not(rgb_2_bw)
cv2.imshow('img',img_invert)
cv2.waitKey(5000)


gaussian_blur=cv2.GaussianBlur(img_invert, (5, 5), 0)
cv2.imshow('blur_img',gaussian_blur)
cv2.waitKey(5000)


final = cv2.divide(rgb_2_bw, 255 - gaussian_blur, scale=255)
cv2.imshow('sketch',final)
cv2.waitKey(5000)

cv2.imwrite('davis_sketch.jpg', final)
sketch=cv2.imread('davis_sketch.jpg')

print(final.shape)
print(sketch.shape)

combined=cv2.hconcat([resized_img,sketch])
cv2.imshow('combined',combined)
cv2.waitKey(5000)
cv2.imwrite('davis_sketch_combined.jpg', combined)

#print(combined.shape)
