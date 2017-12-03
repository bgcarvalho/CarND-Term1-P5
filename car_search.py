import numpy as np
import cv2
from lane_search import Lane
from skimage.feature import hog


class Car():

    def __init__(self, model=None, scaler=None):
        self.n_frame = 0
        self.lanes = Lane(monitor=False)
        self.model = model
        self.scaler = scaler

        self.orient = 9
        self.window = 64
        self.spatial_size = (16, 16)
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hist_bins = 32
        self.cells_per_step = 2
        self.color_space = 'LUV'
        self.found_car_box_color = (0, 0, 0)
        # self.found_car_box_color = (0, 0, 255)
        # self.found_car_box_color = (255, 0, 0)

        self.current_boxes = []

    def processFrame(self, img):
        self.n_frame += 1

        ysal = [400, 410, 420, 440, 440]
        ysol = [656, 650, 650, 650, 650]
        scales = [1, 1.25, 1.5, 1.75, 2]

        if self.n_frame % 10 == 0:
            self.current_boxes = []
            for ystart, ystop, scale in zip(ysal, ysol, scales):
                out_img, box_list = find_cars(img,
                                              self.n_frame,
                                              self.color_space,
                                              ystart,
                                              ystop,
                                              scale,
                                              self.model,
                                              self.scaler,
                                              self.orient,
                                              self.pix_per_cell,
                                              self.cell_per_block,
                                              self.cells_per_step,
                                              self.spatial_size,
                                              self.window,
                                              self.hist_bins)
                self.current_boxes.append(box_list)

            car_bbox = self.apply_heat(img, box_list)
            out_img = self.draw_locations(car_bbox)
        else:
            # draw_locations
            pass

        # =====================
        # Find Road Lane Lines
        # =====================
        out_img = self.lanes.searchFrame(out_img)

        return out_img

    def draw_locations(self, img, bboxes, thick=6, form='box'):
        # make a copy of the image
        draw_img = np.copy(img)
        if form == 'box':
            # draw each bounding box on your image copy using cv2.rectangle()
            for box in bboxes:
                color = self.found_car_box_color
                cv2.rectangle(draw_img, box[0], box[1], color, thick)
        elif form == 'circle':
            center = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
            radius = (box[1][0] - box[0][0]) / 2
            color = self.found_car_box_color
            cv2.circle(draw_img, center, radius, color, thick)

        # return the image copy with boxes drawn
        return draw_img

    def color_hist(self, img, nb=32, br=(0, 256)):
        # Compute the histogram of the RGB channels separately
        rhist, bin_edges = np.histogram(img[:, :, 0], bins=nb, range=br, density=True)
        rhist = rhist * np.diff(bin_edges)

        ghist, bin_edges = np.histogram(img[:, :, 1], bins=nb, range=br, density=True)
        ghist = ghist * np.diff(bin_edges)

        bhist, bin_edges = np.histogram(img[:, :, ], bins=nb, range=br, density=True)
        bhist = bhist * np.diff(bin_edges)

        # Concatenate the histograms into a single feature vector
        # hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        hist_features = np.concatenate((rhist, ghist, bhist))

        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def bin_spatial(self, img, size=(32, 32)):
        features = cv2.resize(img, size).ravel()
        if np.amax(features) > 1:
            features = features * 1. / 255
        return features

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis is True:
            features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def find_cars(self, img, i, cs, ystart, ystop, scale, svc, X_scaler,
                  orient, pix_per_cell, cell_per_block, cps, spatial_size,
                  window, hist_bins):
        draw_img = np.copy(img)
        # convert JPG [0 - 255] to PNG [0.0 - 1.0] scale
        img = img.astype(np.float32) / 255.0
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch, conv=cs)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            newsize = (np.int(imshape[1] / scale), np.int(imshape[0] / scale))
            ctrans_tosearch = cv2.resize(ctrans_tosearch, newsize)

        # color channels
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        # nfeat_per_block = orient * cell_per_block ** 2

        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = cps
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire search region
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell,
                                     cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell,
                                     cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell,
                                     cell_per_block, feature_vec=False)

        # boxlist = []
        bbox = []
        for xb in range(nxsteps):
            for yb in range(nysteps):

                # walk in cells
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.ravel([hog_feat1, hog_feat2, hog_feat3])

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window,
                                    xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)
                features = np.concatenate((spatial_features,
                                           hist_features,
                                           hog_features)).reshape(1, -1)
                test_features = X_scaler.transform(features)
                test_prediction = svc.predict(test_features)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                p1 = (xbox_left, ytop_draw + ystart)
                p2 = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                # boxlist.append((p1,p2))
                t = 6

                # cv2.rectangle(draw_img, (xleft, ytop + ystart), (xleft + window, ytop + window + ystart), (0, 0, 0), 5)
                # cv2.rectangle(draw_img, p1, p2, (255, 255, 255), 2)  # white box
                if test_prediction == 1:
                    # blue
                    cor = self.found_car_box_color
                    cv2.rectangle(draw_img, p1, p2, cor, t)
                    bbox.append((p1,p2))
                else:
                    # green
                    cor = (0, 255, 0)
                    # cv2.rectangle(draw_img, p1, p2, cor, 3) 

        return draw_img, bbox

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def draw_labeled_boxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            blue = (0, 0, 255)
            cv2.rectangle(img, bbox[0], bbox[1], blue, 6)

        return img
