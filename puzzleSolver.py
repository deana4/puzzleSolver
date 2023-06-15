import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

class SiftExtractor:
    def __init__(self):
        self.classSift = cv2.SIFT_create()  #creating sift xfeature object

    def extractFeaturesAndDescriptors(self, image):  #this function extract the keypoints and the descriptors using SIFT algo and return it with the extend of the original image drawed the points on it.
        keypoints, descriptors = self.classSift.detectAndCompute(image, None)
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        return (keypoints, descriptors, img_with_keypoints)

class PackagePhotos:
    def __init__(self):
        self.siftManager = SiftExtractor()
        self.imageList = []
        self.perspectiveMat = None
        self.row = None
        self.col = None

    def extractPhotosFromPath(self, path): #path exmaple puzzles\puzzle_affine_1\pieces
        self.imageList.clear()
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.imageList.append((grayImg, img))
        return self.imageList

    def siftPuzzle(self):
        imagesDetailsList = []
        for i in range(len(self.imageList)):
            imageGray, imageColored = self.imageList[i]
            if i == 0:
                imageGray = fixPerspective(imageGray, self.perspectiveMat, self.col, self.row)
                imagesDetailsList.append(self.siftManager.extractFeaturesAndDescriptors(imageGray))
            else:
                imagesDetailsList.append(self.siftManager.extractFeaturesAndDescriptors(imageGray))

        return imagesDetailsList

    def setPerspectiveMat(self, mat, W, H):
        self.perspectiveMat = mat
        self.col = W
        self.row = H

def getFinalMatches(desc1, desc2):
    #step2
    matForPicOne = KNN(desc1, desc2)
    #step3
    matchIndexesFirst, matchIndexesSecond = extractMatchFromMat(matForPicOne)
    #distance, index_point, itsMatch_index - format
    return ratioTest(matchIndexesFirst, matchIndexesSecond)

def checkFunc(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test to keep only good matches
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            goodMatches.append(m)

    return goodMatches

def KNN(desc1, desc2):
    row = len(desc1)
    col = len(desc2)

    matchMatrix = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            matchMatrix[i, j] = np.linalg.norm(desc1[i]-desc2[j])

    return matchMatrix

def extractMatchFromMat(matchMatrix):
    row, col = matchMatrix.shape
    matchIndexesFirst = []
    matchIndexesSecond = []
    for i in range(row):
        minIndex = np.argmin(matchMatrix[i])
        minVal = min(matchMatrix[i])
        matchIndexesFirst.append((minVal, minIndex))
        matchMatrix[i][minIndex] = np.inf

    for i in range(row):
        minIndex = np.argmin(matchMatrix[i])
        minVal = min(matchMatrix[i])
        matchIndexesSecond.append((minVal, minIndex))

    return matchIndexesFirst, matchIndexesSecond

def ratioTest(matchesArr1, matchesArr2):
    finalMatches = []
    loopVar = len(matchesArr1)
    for i in range(loopVar):
        distance_1, index_1 = matchesArr1[i]
        distance_2, index_2 = matchesArr2[i]

        if distance_1 < 0.8*distance_2:
            finalMatches.append((distance_1, i, index_1))  #?  point(index_i) -> point(index_1)

    return finalMatches

def RandomMatches(rangeOfNumbers, affine = True):
    # np.random.seed(2)
    if affine:
        return np.random.randint(rangeOfNumbers, size=3)
    else:
        return np.random.randint(rangeOfNumbers, size=4)

def getTransform(listOfMatchingPoints, isAffine = True):
    if len(listOfMatchingPoints) == 0:
        return "Error Reading ListOfMatchingPoint - Size(0)"

    srcPoints = []
    dstPoints = []
    for point, itsMatch in listOfMatchingPoints:
        srcPoints.append(point)
        dstPoints.append(itsMatch)


    srcPoints = np.float32(srcPoints)
    dstPoints = np.float32(dstPoints)


    if len(listOfMatchingPoints) != 0:
        if isAffine:
            M, mask = cv2.estimateAffine2D(srcPoints, dstPoints)
            return M
        else:
            M, mask = cv2.findHomography(srcPoints, dstPoints)
            return M

def findTheBestHomographicTransform(affineDesc1, affineDesc2, listOfPhotos, index1, indexOfBestMatch, distThreshold, ransacLoop, isAffine = False):

    print("Finding Homographic Transform")

    maxTrans = np.zeros((4, 4))
    listOfMatchingPoints = []

    finalMatches = getFinalMatches(affineDesc1, affineDesc2)

    if len(finalMatches) == 0:
        return maxTrans, 0


    for i in range(len(finalMatches)):
        distance, srcIndex, DstIndex = finalMatches[i]

        point = listOfPhotos[indexOfBestMatch][0][srcIndex]
        itsMatch = listOfPhotos[index1][0][DstIndex]

        listOfMatchingPoints.append([point.pt, itsMatch.pt])

    #RANSAC LOOP
    Maxinliers = 0

    for i in range(ransacLoop):
        inliers = 0
        listOfRandomIndexes = RandomMatches(len(finalMatches), isAffine)

        pointsForM = []
        for rand in listOfRandomIndexes:
            pointsForM.append(listOfMatchingPoints[rand])


        M = getTransform(pointsForM, isAffine)

        toActiveMOn = []
        for point, itsMatch in listOfMatchingPoints:
            toActiveMOn.append(np.float32(point))

        if M is None:
            i = i + 1
            continue

        dst = cv2.perspectiveTransform(np.float32(toActiveMOn).reshape(-1, 1, 2), M)

        for i in range(len(dst)):
            dist = np.linalg.norm(dst[i][0] - listOfMatchingPoints[i][1])
            if dist < distThreshold:
                inliers += 1

        if inliers > Maxinliers:
            Maxinliers = inliers
            maxTrans = M

    return maxTrans, Maxinliers

def findTheBestAffineTransform(affineDesc1, affineDesc2, listOfPhotos, index1, indexOfBestMatch, distThreshold, ransacLoop, isAffine = True):

    print("Finding Affine Transform")

    maxTrans = np.zeros((3, 3))
    listOfMatchingPoints = []


    finalMatches = getFinalMatches(affineDesc1, affineDesc2)

    if len(finalMatches) == 0:
        return maxTrans, 0

    for i in range(len(finalMatches)):
        distance, srcIndex, DstIndex = finalMatches[i]

        point = listOfPhotos[indexOfBestMatch][0][srcIndex]
        itsMatch = listOfPhotos[index1][0][DstIndex]

        listOfMatchingPoints.append([point.pt, itsMatch.pt])

    #RANSAC LOOP
    Maxinliers = 0

    for i in range(ransacLoop):
        inliers = 0
        listOfRandomIndexes = RandomMatches(len(finalMatches), isAffine)
        pointsForM = []
        for rand in listOfRandomIndexes:
            pointsForM.append(listOfMatchingPoints[rand])


        M = getTransform(pointsForM, isAffine)

        toActiveMOn = []
        for point, itsMatch in listOfMatchingPoints:
            toActiveMOn.append(np.float32(point))

        dst = cv2.transform(np.float32(toActiveMOn).reshape(-1, 1, 2), M)


        for i in range(len(dst)):
            dist = np.linalg.norm(dst[i][0] - listOfMatchingPoints[i][1])
            if dist < distThreshold:
                inliers += 1

        if inliers > Maxinliers:
            Maxinliers = inliers
            maxTrans = M

    return maxTrans, Maxinliers

def warpImages(img1, img2, M, row, col, isAffine = True):
    if isAffine == True:
        warp_image2 = cv2.warpAffine(img2, M, (row, col))
    else:
        warp_image2 = cv2.warpPerspective(img2, M, (row, col))

    ret = np.maximum(img1, warp_image2)
    return ret, warp_image2

def fixPerspective(imageToWarp, perspectiveMat, row, col):
    return cv2.warpPerspective(imageToWarp, perspectiveMat, (row, col))

def saveImagesFromArray(listOfImages, directoryName, imageID):
    directory = directoryName + "/PartialPartsForPuzzle" + str(imageID)
    os.mkdir(directory)
    i = 1
    for image, pieceID in listOfImages:
        cv2.imwrite(directory + "/relative_" + str(pieceID) + "_piece.jpg", image)
        i = i + 1

def makeCoverageMap(listOfRelativeLocation, directoryName):

    gray_images = []
    masks = []
    for image, k in listOfRelativeLocation:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        gray_images.append(gray)
        masks.append(mask)

    overlapping_pixels_count = np.zeros_like(gray_images[0], dtype=np.float32)

    for i in range(overlapping_pixels_count.shape[0]):
        for j in range(overlapping_pixels_count.shape[1]):
            overlapping_count = 0
            for mask in masks:
                if mask[i][j] > 0:
                    overlapping_count += 1
            overlapping_pixels_count[i][j] = overlapping_count

    max = 0
    for i in range(overlapping_pixels_count.shape[0]):
        for j in range(overlapping_pixels_count.shape[1]):
            if overlapping_pixels_count[i][j] > max:
                max = overlapping_pixels_count[i][j]

    plt.imshow(overlapping_pixels_count, vmin=0, vmax=max)
    plt.colorbar()

    output_path = os.path.join(directoryName, "overlapping_pixels_count.png")
    plt.savefig(output_path)

    #plt.show()
    plt.close()

def getPuzzle(imageID, directoryName, siftImages, path, perspectiveMatPath, W, H, distThreshold = 7, inlierTreshold = 10, ransacLoop = 1000, isAffine = True):
    numberOfPartsConnected = 1
    matrix = np.loadtxt(perspectiveMatPath)
    print("Runsac loop var", ransacLoop)

    listOfPhotosArray = siftImages.extractPhotosFromPath(path)
    siftImages.setPerspectiveMat(matrix, W, H)
    listOfPhotos = siftImages.siftPuzzle()
    listOfRelativeLocation = []
    numOfPieces = len(listOfPhotosArray)


    matrix = np.loadtxt(perspectiveMatPath)
    f1 = fixPerspective(listOfPhotosArray[0][0], matrix, W, H)
    f2 = fixPerspective(listOfPhotosArray[0][1], matrix, W, H)
    listOfPhotosArray[0] = (f1, f2)

    listOfRelativeLocation.append((listOfPhotosArray[0][1], 1))

    k = numOfPieces
    # imgplot = plt.imshow(listOfPhotosArray[0][1])
    # plt.show()

    while k != 1:
        print("Piece Number", k)
        inliersMatchMat = np.zeros(numOfPieces) - 1

        firstPhotoDesc = listOfPhotos[0][1]


        print("finding the next piece")
        for i in range(1, len(listOfPhotosArray)):
            if listOfPhotosArray[i] != -1:
                finMatches = checkFunc(firstPhotoDesc, listOfPhotos[i][1])
                if len(finMatches) == 0:
                    inliersMatchMat[i] = -1
                else:
                    inliersMatchMat[i] = len(finMatches)


        indxBestMatch = np.argmax(inliersMatchMat)
        if (inliersMatchMat[indxBestMatch] == -1) | (inliersMatchMat[indxBestMatch] == 0):
            print("Finished Parts")
            break

        if isAffine:
            if inliersMatchMat[indxBestMatch] < 3:
                print("Skipping part")
                listOfPhotosArray[indxBestMatch] = -1
                k = k - 1
                continue
        else:
            if inliersMatchMat[indxBestMatch] < 4:
                print("Skipping part")
                listOfPhotosArray[indxBestMatch] = -1
                k = k - 1
                continue

        print("connecting pieces")
        if isAffine:
            maxTrans, maxInliers = findTheBestAffineTransform(listOfPhotos[indxBestMatch][1], firstPhotoDesc, listOfPhotos, 0, indxBestMatch, distThreshold, ransacLoop, isAffine)
        else:
            maxTrans, maxInliers = findTheBestHomographicTransform(listOfPhotos[indxBestMatch][1], firstPhotoDesc, listOfPhotos, 0, indxBestMatch, distThreshold, ransacLoop, isAffine)

        #print(inliersMatchMat, indxBestMatch, inliersMatchMat[indxBestMatch], maxInliers)

        if maxInliers < inlierTreshold:
            print("Number of inliers is low")
            listOfPhotosArray[indxBestMatch] = -1
            k = k - 1
            continue

        if isAffine == True:
            _row, _col = maxTrans.shape
            if _row * _col != 6:
                listOfPhotosArray[indxBestMatch] = -1
                k = k - 1
                print("Mat shapes arent correct")
                continue

        elif isAffine == False:
            _row, _col = maxTrans.shape
            if _row * _col != 9:
                listOfPhotosArray[indxBestMatch] = -1
                k = k - 1
                print("Mat shapes arent correct")
                continue

        print("piece connected successfully")
        numberOfPartsConnected = numberOfPartsConnected + 1
        img1 = listOfPhotosArray[0][1]
        img2 = listOfPhotosArray[indxBestMatch][1]
        imgWrapped, relativeLocationImage = warpImages(img1, img2, maxTrans, W, H, isAffine)
        listOfRelativeLocation.append((relativeLocationImage, k))

        # imgplot = plt.imshow(imgWrapped)
        # plt.show()

        #Updates
        listOfPhotosArray[0] = (cv2.cvtColor(imgWrapped, cv2.COLOR_BGR2GRAY), imgWrapped)
        listOfPhotos[0] = siftImages.siftManager.extractFeaturesAndDescriptors(listOfPhotosArray[0][1])
        listOfPhotosArray[indxBestMatch] = -1

        k = k - 1

    saveImagesFromArray(listOfRelativeLocation, directoryName, imageID)
    makeCoverageMap(listOfRelativeLocation, directoryName)

    return listOfPhotosArray[0][1], numberOfPartsConnected, numOfPieces


if __name__ == '__main__':
    # startTime = time.time()
    listOfPuzzlePartsPathsAffine = [("puzzles/puzzle_affine_1/pieces", "puzzles/puzzle_affine_1/warp_mat_1__H_521__W_760_.txt", 760, 521, 1, 10, 1000, True),
                                    ("puzzles/puzzle_affine_2/pieces", "puzzles/puzzle_affine_2/warp_mat_1__H_537__W_735_.txt", 735, 537, 2, 10, 1000, True),
                                    ("puzzles/puzzle_affine_3/pieces", "puzzles/puzzle_affine_3/warp_mat_1__H_497__W_741_.txt", 741, 497, 2, 10, 1000, True),
                                    ("puzzles/puzzle_affine_4/pieces", "puzzles/puzzle_affine_4/warp_mat_1__H_457__W_808_.txt", 808, 457, 1, 5, 1000, True),
                                    ("puzzles/puzzle_affine_5/pieces", "puzzles/puzzle_affine_5/warp_mat_1__H_510__W_783_.txt", 783, 510, 2, 10, 1000, True),
                                    ("puzzles/puzzle_affine_6/pieces", "puzzles/puzzle_affine_6/warp_mat_1__H_522__W_732_.txt", 732, 522, 2, 10, 3000, True),
                                    ("puzzles/puzzle_affine_7/pieces","puzzles/puzzle_affine_7/warp_mat_1__H_511__W_732_.txt", 732, 511, 2, 5, 4000, True),
                                    ("puzzles/puzzle_affine_8/pieces", "puzzles/puzzle_affine_8/warp_mat_1__H_457__W_811_.txt", 811, 457, 1.3, 5, 3500, True),
                                    ("puzzles/puzzle_affine_9/pieces", "puzzles/puzzle_affine_9/warp_mat_1__H_481__W_771_.txt", 771, 481, 0.5, 5, 3000, True),
                                    ("puzzles/puzzle_affine_10/pieces", "puzzles/puzzle_affine_10/warp_mat_1__H_507__W_771_.txt", 771, 507, 5, 6, 3000, True)]
    listOfPuzzlePartsPathsHomography = [("puzzles/puzzle_homography_1/pieces","puzzles/puzzle_homography_1/warp_mat_1__H_549__W_699_.txt", 699, 549, 2, 5,3000, False),
                                        ("puzzles/puzzle_homography_2/pieces","puzzles/puzzle_homography_2/warp_mat_1__H_513__W_722_.txt",722, 513, 2, 5, 3000, False),
                                        ("puzzles/puzzle_homography_3/pieces","puzzles/puzzle_homography_3/warp_mat_1__H_502__W_760_.txt", 760, 502, 2, 5,3000, False),
                                        ("puzzles/puzzle_homography_4/pieces","puzzles/puzzle_homography_4/warp_mat_1__H_470__W_836_.txt", 836,470, 2, 5, 3000, False),
                                        ("puzzles/puzzle_homography_5/pieces","puzzles/puzzle_homography_5/warp_mat_1__H_457__W_811_.txt",811, 457, 0.5, 5, 3000, False),
                                        ("puzzles/puzzle_homography_6/pieces","puzzles/puzzle_homography_6/warp_mat_1__H_464__W_815_.txt", 815, 464, 1, 5,10000, False), 
                                        ("puzzles/puzzle_homography_7/pieces","puzzles/puzzle_homography_7/warp_mat_1__H_488__W_760_.txt", 760,488, 1, 5, 10000, False), 
                                        ("puzzles/puzzle_homography_8/pieces","puzzles/puzzle_homography_8/warp_mat_1__H_499__W_760_.txt",760, 499, 1.5, 4, 10000, False), 
                                        ("puzzles/puzzle_homography_9/pieces","puzzles/puzzle_homography_9/warp_mat_1__H_490__W_816_.txt", 816, 490, 15, 13,20000, False), 
                                        ("puzzles/puzzle_homography_10/pieces","puzzles/puzzle_homography_10/warp_mat_1__H_506__W_759_.txt",759, 506, 1.5, 10, 17000, False)]
    listOfSuccessfulParts = []
    siftImages = PackagePhotos()

    imageID = 1

    for pathOfPuzzleParts, PathOfMat, W, H, distTreshold, maxInlierTrashold, numOfRansacIterations, isAffine in listOfPuzzlePartsPathsAffine:
        directoryName = 'Solution_Puzzle_Affine' + str(imageID)
        if os.path.exists(directoryName):
            shutil.rmtree(directoryName)
        os.makedirs(directoryName)

        completePuzzle, successfulParts, totalAmountOfPieces = getPuzzle(imageID, directoryName, siftImages, pathOfPuzzleParts, PathOfMat, W, H, distTreshold, maxInlierTrashold, numOfRansacIterations, isAffine)
        fileName = "solution_" + str(successfulParts) + "_" + str(totalAmountOfPieces) + ".jpeg"
        cv2.imwrite(os.path.join(directoryName, fileName), completePuzzle)
        listOfSuccessfulParts.append(successfulParts)
        imageID = imageID + 1

    imageID = 1

    for pathOfPuzzleParts, PathOfMat, W, H, distTreshold, maxInlierTrashold, numOfRansacIterations, isAffine in listOfPuzzlePartsPathsHomography:
        directoryName = 'Solution_Puzzle_Homographic' + str(imageID)
        if os.path.exists(directoryName):
            shutil.rmtree(directoryName)
        os.makedirs(directoryName)

        completePuzzle, successfulParts, totalAmountOfPieces = getPuzzle(imageID, directoryName, siftImages, pathOfPuzzleParts, PathOfMat, W, H, distTreshold, maxInlierTrashold, numOfRansacIterations, isAffine)
        fileName = "solution_" + str(successfulParts) + "_" + str(totalAmountOfPieces) + ".jpeg"
        cv2.imwrite(os.path.join(directoryName, fileName), completePuzzle)
        listOfSuccessfulParts.append(successfulParts)
        imageID = imageID + 1



    # print(listOfSuccessfulParts)
    # print("Total Time To Finish" +  str((time.time() - startTime)/60) + "Minutes")
