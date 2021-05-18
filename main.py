from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse


def generate_tiles(wsi_path):
    return 'tile_path'


def match_cell_to_tile(tile_path, cell_path):
    """
    Attempts to match an image of a cell to a single tile from a WSI.

    Parameters
        tile_path: str
            The path to the tile image
        cell_path: str
            The path to the image of the cell
    """
    tile_image = cv.imread(tile_path, cv.IMREAD_GRAYSCALE)
    cell_image = cv.imread(cell_path, cv.IMREAD_GRAYSCALE)

    # Detect  keypoints using SURF
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(tile_image, None)
    keypoints2, descriptors2 = detector.detectAndCompute(cell_image, None)

    # Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    # Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []

    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = np.empty((max(tile_image.shape[0], cell_image.shape[0]), tile_image.shape[1]+cell_image.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(tile_image, keypoints1, cell_image, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show detected matches
    cv.imshow('Good Matches', img_matches)
    cv.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for matching cell images to WSIs.')
    parser.add_argument('--wsi', help='Path to whole slide image')
    parser.add_argument('--cells', help='Path to cell images')
    args = parser.parse_args()
    tile_path = generate_tiles(args.wsi)
    match_cell_to_tile(tile_path, args.cells)