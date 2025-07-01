import cv2 

# load the image
image = cv2.imread("code/Elephant2.jpg")

# Create a copy to avoind modifying the original image
image_copy = image.copy()

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle on the image copy
        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
        # Print the coordinates
        print(f"Point selected: ({x}, {y})")
        # Show the updated image
        cv2.imshow("Image", image_copy)


# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_circle)
# Display the image
cv2.imshow("Image", image_copy)
# Wait for a key press
cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()

