import cv2
import numpy as np
import os

def get_sift_matches(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use a FLANN based matcher
    matcher = cv2.FlannBasedMatcher()

    # Perform KNN matching from img1 to img2
    matches_img1_to_img2 = matcher.knnMatch(des1, des2, k=2)

    # Perform KNN matching from img2 to img1
    matches_img2_to_img1 = matcher.knnMatch(des2, des1, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches_img1_to_img2:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            # Check if the reverse match is consistent with the original match(cross check)
            reverse_match = matches_img2_to_img1[m.trainIdx][0]
            if reverse_match.trainIdx == m.queryIdx:
                good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return kp1, kp2, good_matches, src_pts, dst_pts

def apply_ransac(src_pts, dst_pts):
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()

    return mask

def calculate_similarity_score(filtered_matches, matches1, matches2):
    num_keypoints1 = len(matches1)
    num_keypoints2 = len(matches2)
    filtered_matches = len(filtered_matches)
    similarity_score = filtered_matches / min(num_keypoints1, num_keypoints2)
    return similarity_score



def sift_feature_matching(image1_path, image2_path):
    # Read images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Get SIFT matches
    kp1, kp2, good_matches, src_pts, dst_pts = get_sift_matches(img1, img2)

    # Apply RANSAC to filter outliers
    mask = apply_ransac(src_pts, dst_pts)

    # Filter matches based on RANSAC mask
    ransac_matches = [match for i, match in enumerate(good_matches) if mask[i] == 1]

    # Check if matches are valid
    if good_matches is None:
        return

    # Calculate similarity score
    similarity_score = len(ransac_matches) / min(len(kp1), len(kp2))
    print(f"Similarity Score between {image1_path} and {image2_path}: {similarity_score}")

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, ransac_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Determine if images are similar based on a threshold
    if similarity_score > 0.5:
        print("Images are similar.")
    else:
        print("Images are not similar.")

if __name__ == "__main__":

    image_pairs = [
        ("image1a.jpeg", "image1b.jpeg"),
        ("image2a.jpeg", "image2b.jpeg"),
        ("image3a.jpeg", "image3b.jpeg"),
        ("image4a.jpeg", "image4b.jpeg"),
        ("image4a.jpeg", "image4c.png"),
        ("image4b.jpeg", "image4c.png"),
        ("image5a.jpeg", "image5b.jpeg"),
        ("image6a.jpeg", "image6b.jpeg"),
        ("image7a.jpeg", "image7b.jpeg")
    ]


    for image1_path, image2_path in image_pairs:
        sift_feature_matching(image1_path, image2_path)
